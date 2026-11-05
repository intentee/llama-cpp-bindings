use crate::parsed_expected_failure::ParsedExpectedFailure;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ParsedExpectedOutcome {
    Success,
    Failure(ParsedExpectedFailure),
}
