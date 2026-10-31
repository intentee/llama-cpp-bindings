use enumflags2::bitflags;

#[derive(Eq, PartialEq, Debug, Clone, Copy)]
#[bitflags]
#[repr(u8)]
pub enum MtmdCapability {
    Chat = 1 << 0,
    Vision = 1 << 1,
    Audio = 1 << 2,
    Video = 1 << 3,
    AudioStream = 1 << 4,
    TextGeneration = 1 << 5,
    AudioGeneration = 1 << 6,
    AudioGenerationSpeakerReference = 1 << 7,
}
