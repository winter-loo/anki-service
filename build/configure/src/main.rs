// Copyright: Ankitects Pty Ltd and contributors
// License: GNU AGPL, version 3 or later; http://www.gnu.org/licenses/agpl.html

mod platform;
mod proto;
mod pylib;
mod python;
mod rust;

use ninja_gen::python::setup_python;
use ninja_gen::Build;
use ninja_gen::Result;
use pylib::build_pylib;
use pylib::check_pylib;
use python::check_copyright;
use python::check_python;
use python::setup_venv;
use rust::build_rust;
use rust::check_rust;

use crate::proto::check_proto;


fn main() -> Result<()> {
    let mut build = Build::new()?;
    let build = &mut build;

    setup_python(build)?;
    setup_venv(build)?;

    build_rust(build)?;
    build_pylib(build)?;

    check_rust(build)?;
    check_pylib(build)?;
    check_python(build)?;
    check_proto(build)?;
    check_copyright(build)?;

    build.trailing_text = "default pylib/anki\n".into();

    build.write_build_file();

    Ok(())
}
