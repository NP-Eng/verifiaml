pub(crate) mod model;
pub(crate) mod quantization;
pub(crate) mod utils;

#[cfg(test)]
pub(crate) mod pcs_types;

trait Commitment {}

trait CommitmentState {}

trait Proof {}
