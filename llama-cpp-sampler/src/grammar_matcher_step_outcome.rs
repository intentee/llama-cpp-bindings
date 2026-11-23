pub enum GrammarMatcherStepOutcome<TValue> {
    Produced(TValue),
    BenignStop,
}
