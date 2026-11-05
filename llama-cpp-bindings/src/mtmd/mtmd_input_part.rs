use super::mtmd_bitmap::MtmdBitmap;
use super::mtmd_input_text_part::MtmdInputTextPart;

#[derive(Clone, Debug)]
pub enum MtmdInputPart<'bitmap> {
    Bitmap(&'bitmap MtmdBitmap),
    Text(MtmdInputTextPart),
}

#[cfg(test)]
mod tests {
    use std::thread;

    use crate::mtmd::mtmd_input_part::MtmdInputPart;
    use crate::mtmd::mtmd_input_text_part::MtmdInputTextPart;

    #[test]
    fn text_parts_move_to_another_thread() {
        let parts = [MtmdInputPart::Text(MtmdInputTextPart {
            text: "hello".to_owned(),
            parse_special: true,
        })];

        assert_eq!(thread::spawn(move || parts.len()).join().ok(), Some(1));
    }
}
