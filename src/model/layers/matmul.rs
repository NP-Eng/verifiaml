use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;

use crate::model::{Layer, Poly};

/// Start with 2D matrices, and Mat-by-vector multiplication only
pub struct MatMulLayer<F> {
    /// The flattened vector of weights
    pub weights: Vec<F>,
    /// The vector of biases
    pub biases: Vec<F>,
    /// Dimensions (rows, columns)
    pub dims: (usize, usize),
}

impl<F, S, PCS> Layer<F, S, PCS> for MatMulLayer<F>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    type Commitment = PCS::Commitment;

    // this will be the sumcheck proof
    type Proof = PCS::Proof;

    /// Returns the number of nodes in the layer
    fn num_nodes(&self) -> usize {
        self.dims.1
    }

    /// Evaluate the layer on the given input natively.
    fn evaluate(&self, input: Vec<F>) -> Vec<F> {
        assert_eq!(input.len(), self.dims.0);

        let mut output = vec![F::zero(); self.dims.1];
        for i in 0..self.dims.1 {
            for j in 0..self.dims.0 {
                output[i] += input[j] * self.weights[i * self.dims.0 + j];
            }
            output[i] += self.biases[i];
        }
        output
    }

    /// Evaluate the layer on the given input natively.
    fn commit(&self) -> PCS::Commitment {
        unimplemented!()
    }

    /// Prove that the layer was executed correctly on the given input.
    fn prove(com: Self::Commitment, input: Vec<F>) -> Self::Proof {
        unimplemented!()
    }

    /// Check that the layer transition was executed correctly.
    fn check(com: Self::Commitment, proof: Self::Proof) -> bool {
        unimplemented!()
    }
}
