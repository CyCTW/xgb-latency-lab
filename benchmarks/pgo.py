"""Train IR instrumentation profiles on calibration rows only."""
import json
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

def build_pgo(command, entry, out, harness, calibration, *, training_interfaces=None):
    out.mkdir(parents=True, exist_ok=False)
    base = list(command)
    position = base.index("-o")
    del base[position:position+2]
    extension = ".dylib" if sys.platform == "darwin" else ".so"
    instrumented = out / ("instrumented" + extension)
    profile_dir = out / "profiles"
    profile_dir.mkdir()
    generate = [*base, "-fprofile-generate", "-o", str(instrumented)]
    subprocess.run(generate, check=True, capture_output=True)
    # Each native invocation is a new process. LLVM_PROFILE_FILE stays confined
    # to this child and every training row comes from calibration, never tuning.
    data = out / "calibration.f32"
    calibration.tofile(data)
    interfaces = [entry["prepared"]] if training_interfaces is None else training_interfaces
    training_runs = []
    for prepared in interfaces:
        env = dict(os.environ, LLVM_PROFILE_FILE=str(profile_dir / (str(int(prepared))+"-%m.profraw")))
        args = [str(harness), str(data), str(len(calibration)), str(calibration.shape[1]),
                str(max(8192, len(calibration))), "1", "1729", "profile_training",
                str(instrumented), "predict" if prepared else entry["symbol"], str(int(prepared))]
        trained = subprocess.run(args, env=env, check=True, capture_output=True, text=True)
        training_runs.append(json.loads(trained.stdout))
    (out / "instrumentation-run.json").write_text(json.dumps(training_runs, indent=2))
    profiles = sorted(profile_dir.glob("*.profraw"))
    if not profiles:
        raise RuntimeError("PGO training did not produce any raw profile")
    profdata = shutil.which("llvm-profdata")
    if profdata is None and sys.platform == "darwin":
        profdata = subprocess.check_output(["xcrun", "--find", "llvm-profdata"], text=True).strip()
    if profdata is None:
        raise RuntimeError("Install llvm-profdata matching the Clang compiler")
    profile = out / "model.profdata"
    subprocess.run([profdata, "merge", "-o", str(profile), *map(str, profiles)],
                   check=True, capture_output=True)
    summary = subprocess.check_output([profdata, "show", str(profile)], text=True)
    (out / "profile-summary.txt").write_text(summary)
    match = re.search(r"Maximum function count:\s*(\d+)", summary)
    if match is None or int(match[1]) == 0:
        raise RuntimeError("PGO profile has no observed function executions")
    library = out / ("model" + extension)
    use = [*base, "-fprofile-use=" + str(profile), "-Werror=profile-instr-out-of-date",
           "-Werror=profile-instr-unprofiled", "-o", str(library)]
    compiled = subprocess.run(use, check=True, capture_output=True, text=True)
    (out / "compile-stderr.txt").write_text(compiled.stderr)
    if entry["family"] == "engine":
        object_base = [arg for arg in base if arg not in ("-dynamiclib", "-shared")]
        for flags, filename in ((["-c"], "model.o"), (["-S"], "model.s"),
                                (["-S", "-emit-llvm"], "model.opt.ll")):
            subprocess.run([*object_base, *flags, "-fprofile-use=" + str(profile),
                            "-Werror=profile-instr-out-of-date", "-Werror=profile-instr-unprofiled",
                            "-o", str(out / filename)], check=True, capture_output=True)
        shutil.copy2(Path(entry["library"]).parent / "model.h", out / "model.h")
    source_metadata = Path(entry["library"]).parent / "metadata.json"
    metadata = json.loads(source_metadata.read_text()) if source_metadata.exists() else {}
    metadata.update(pgo=True, profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(), library_bytes=library.stat().st_size,
                    profile_training="calibration only", training_interfaces=interfaces,
                    generate_command=generate, use_command=use)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return library, metadata
