// Copyright: Ankitects Pty Ltd and contributors
// License: GNU AGPL, version 3 or later; http://www.gnu.org/licenses/agpl.html

use std::collections::HashMap;

use fluent::FluentArgs;
use fluent::FluentValue;

use super::Backend;
use crate::pb;
pub(super) use crate::pb::i18n::i18n_service::Service as I18nService;
use crate::prelude::*;
use crate::scheduler::timespan::answer_button_time;
use crate::scheduler::timespan::time_span;

impl I18nService for Backend {
    fn translate_string(
        &self,
        input: pb::i18n::TranslateStringRequest,
    ) -> Result<pb::generic::String> {
        let input_str = format!("{:?}", input);
        let args = build_fluent_args(input.args);

        let rst = Ok(self
            .tr
            .translate_via_index(
                input.module_index as usize,
                input.message_index as usize,
                args,
            )
            .into());

        use std::fs::OpenOptions;
        use std::io::Write;
        use std::path::Path;
        use std::time::{SystemTime, UNIX_EPOCH};

        let since_the_epoch = SystemTime::now().duration_since(UNIX_EPOCH).expect("");
        let now_ms = since_the_epoch.as_millis();
        let log = format!(
            "{:?} I18N.TranslateString({:?})={:?}",
            now_ms, input_str, rst
        );

        if Path::new("/Users/ldd/proj/rust/anki/action_study_now").exists() {
            let mut log_file = OpenOptions::new()
                .create(true)
                .append(true)
                .open("service.log")
                .expect("Error opening file");
            if let Err(err) = writeln!(log_file, "{}", log) {
                eprintln!("Error appending to file: {}", err);
            } else {
                println!("string appended to file successfully");
            }
        }

        rst
    }

    fn format_timespan(
        &self,
        input: pb::i18n::FormatTimespanRequest,
    ) -> Result<pb::generic::String> {
        use pb::i18n::format_timespan_request::Context;
        Ok(match input.context() {
            Context::Precise => time_span(input.seconds, &self.tr, true),
            Context::Intervals => time_span(input.seconds, &self.tr, false),
            Context::AnswerButtons => answer_button_time(input.seconds, &self.tr),
        }
        .into())
    }

    fn i18n_resources(&self, input: pb::i18n::I18nResourcesRequest) -> Result<pb::generic::Json> {
        serde_json::to_vec(&self.tr.resources_for_js(&input.modules))
            .map(Into::into)
            .map_err(Into::into)
    }
}

fn build_fluent_args(input: HashMap<String, pb::i18n::TranslateArgValue>) -> FluentArgs<'static> {
    let mut args = FluentArgs::new();
    for (key, val) in input {
        args.set(key, translate_arg_to_fluent_val(&val));
    }
    args
}

fn translate_arg_to_fluent_val(arg: &pb::i18n::TranslateArgValue) -> FluentValue<'static> {
    use pb::i18n::translate_arg_value::Value as V;
    match &arg.value {
        Some(val) => match val {
            V::Str(s) => FluentValue::String(s.to_owned().into()),
            V::Number(f) => FluentValue::Number(f.into()),
        },
        None => FluentValue::String("".into()),
    }
}
