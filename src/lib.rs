use ark_poly::DenseMultilinearExtension;

pub(crate) mod error;
pub(crate) mod hidden_model;
pub(crate) mod model;
pub(crate) mod proofs;
pub(crate) mod qarray;
pub(crate) mod quantization;

trait Commitment {}

trait CommitmentState {}

trait Proof {}

pub(crate) type Poly<F> = DenseMultilinearExtension<F>;
