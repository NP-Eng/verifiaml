use std::marker::PhantomData;

use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use ark_poly_commit::PolynomialCommitment;

use crate::model::nodes::Node;

use self::qarray::QArray;

mod nodes;
mod qarray;

pub(crate) type Poly<F> = DenseMultilinearExtension<F>;

// TODO change the functions that receive vectors to receive slices instead whenever it makes sense

// TODO: for now, we require all layers to use the same PCS; this might change
// in the future
pub struct Model<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    layers: Vec<Node<F, S, PCS>>,
    phantom: PhantomData<(F, S, PCS)>,
}

impl<F, S, PCS> Model<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    
    fn new(layers: Vec<Node<F, S, PCS>>) -> Self {
        Self {
            layers,
            phantom: PhantomData,
        }
    }

    fn evaluate(&self, input: QArray) -> QArray {
        let mut output = input;
        for layer in &self.layers {
            output = layer.evaluate(output);
        }
        output
    }
}
