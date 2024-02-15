use std::error::Error;

#[derive(Debug)]
pub(crate) struct VerificationError(String);

impl Error for VerificationError {}

impl std::fmt::Display for VerificationError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "VerificationError: {}", self.0)
    }
}
