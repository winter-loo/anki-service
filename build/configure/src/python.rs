// Copyright: Ankitects Pty Ltd and contributors
// License: GNU AGPL, version 3 or later; http://www.gnu.org/licenses/agpl.html

use ninja_gen::action::BuildAction;
use ninja_gen::archives::Platform;
use ninja_gen::build::FilesHandle;
use ninja_gen::command::RunCommand;
use ninja_gen::glob;
use ninja_gen::hashmap;
use ninja_gen::input::BuildInput;
use ninja_gen::inputs;
use ninja_gen::python::python_format;
use ninja_gen::python::PythonEnvironment;
use ninja_gen::python::PythonLint;
use ninja_gen::python::PythonTypecheck;
use ninja_gen::rsync::RsyncFiles;
use ninja_gen::Build;
use ninja_gen::Result;

pub fn setup_venv(build: &mut Build) -> Result<()> {
    build.add(
        "pyenv",
        PythonEnvironment {
            folder: "pyenv",
            base_requirements_txt: inputs!["python/requirements.txt"],
            requirements_txt: inputs!["python/requirements.txt"],
            extra_binary_exports: &[
                "pip-compile",
                "pip-sync",
                "mypy",
                "black",
                "isort",
                "pylint",
                "pytest",
                "protoc-gen-mypy",
            ],
        },
    )?;

    Ok(())
}

pub struct GenPythonProto {
    pub proto_files: BuildInput,
}

impl BuildAction for GenPythonProto {
    fn command(&self) -> &str {
        "$protoc $
        --plugin=protoc-gen-mypy=$protoc-gen-mypy $
        --python_out=$builddir/pylib $
        --mypy_out=$builddir/pylib $
        -Iproto $in"
    }

    fn files(&mut self, build: &mut impl FilesHandle) {
        let proto_inputs = build.expand_inputs(&self.proto_files);
        let python_outputs: Vec<_> = proto_inputs
            .iter()
            .flat_map(|path| {
                let path = path
                    .replace('\\', "/")
                    .replace("proto/", "pylib/")
                    .replace(".proto", "_pb2");
                [format!("{path}.py"), format!("{path}.pyi")]
            })
            .collect();
        build.add_inputs("in", &self.proto_files);
        build.add_inputs("protoc", inputs!["$protoc_binary"]);
        build.add_inputs("protoc-gen-mypy", inputs![":pyenv:protoc-gen-mypy"]);
        build.add_outputs("", python_outputs);
    }
}


pub fn check_python(build: &mut Build) -> Result<()> {
    python_format(build, "ftl", inputs![glob!("ftl/**/*.py")])?;
    python_format(build, "tools", inputs![glob!("tools/**/*.py")])?;

    build.add(
        "check:mypy",
        PythonTypecheck {
            folders: &[
                "ftl",
                "python",
                "tools",
            ],
            deps: inputs![glob!["{ftl}/**/*.{py,pyi}"]],
        },
    )?;

    add_pylint(build)?;

    Ok(())
}

fn add_pylint(build: &mut Build) -> Result<()> {
    // pylint does not support PEP420 implicit namespaces split across import paths,
    // so we need to merge our pylib sources and generated files before invoking it,
    // and add a top-level __init__.py
    build.add(
        "pylint/anki",
        RsyncFiles {
            inputs: inputs![":pylib/anki"],
            target_folder: "pylint/anki",
            strip_prefix: "$builddir/pylib/anki",
            // avoid copying our large rsbridge binary
            extra_args: "--links",
        },
    )?;
    build.add(
        "pylint/anki",
        RsyncFiles {
            inputs: inputs![glob!["pylib/anki/**"]],
            target_folder: "pylint/anki",
            strip_prefix: "pylib/anki",
            extra_args: "",
        },
    )?;
    build.add(
        "pylint/anki",
        RunCommand {
            command: ":pyenv:bin",
            args: "$script $out",
            inputs: hashmap! { "script" => inputs!["python/mkempty.py"] },
            outputs: hashmap! { "out" => vec!["pylint/anki/__init__.py"] },
        },
    )?;
    build.add(
        "check:pylint",
        PythonLint {
            folders: &[
                "$builddir/pylint/anki",
                "ftl",
                "pylib/tools",
                "tools",
                "python",
            ],
            pylint_ini: inputs![".pylintrc"],
            deps: inputs![
                ":pylint/anki",
                glob!("{pylib/tools,ftl,python,tools}/**/*.py")
            ],
        },
    )?;

    Ok(())
}

pub fn check_copyright(build: &mut Build) -> Result<()> {
    let script = inputs!["tools/copyright_headers.py"];
    let files = inputs![glob!["{build,rslib,pylib,ftl,python,tools}/**/*.{py,rs}"]];
    build.add(
        "check:copyright",
        RunCommand {
            command: "$runner",
            args: "run --stamp=$out $pyenv_bin $script check",
            inputs: hashmap! {
                "pyenv_bin" => inputs![":pyenv:bin"],
                "script" => script.clone(),
                "script" => script.clone(),
                "" => files.clone(),
            },
            outputs: hashmap! {
                "out" => vec!["tests/copyright.check.marker"]
            },
        },
    )?;
    build.add(
        "fix:copyright",
        RunCommand {
            command: "$runner",
            args: "run --stamp=$out $pyenv_bin $script fix",
            inputs: hashmap! {
                "pyenv_bin" => inputs![":pyenv:bin"],
                "script" => script,
                "" => files,
            },
            outputs: hashmap! {
                "out" => vec!["tests/copyright.fix.marker"]
            },
        },
    )?;
    Ok(())
}
