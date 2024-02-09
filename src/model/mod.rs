use ark_std::marker::PhantomData;

use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use ark_poly_commit::PolynomialCommitment;

use crate::{model::nodes::Node, quantization::QSmallType};

use self::qarray::QArray;

mod examples;
mod nodes;
mod qarray;
mod reshaping;

pub(crate) type Poly<F> = DenseMultilinearExtension<F>;

// TODO change the functions that receive vectors to receive slices instead whenever it makes sense

// TODO: for now, we require all nodes to use the same PCS; this might change
// in the future
pub struct Model<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    nodes: Vec<Node<F, S, PCS>>,
    phantom: PhantomData<(F, S, PCS)>,
}

impl<F, S, PCS> Model<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    pub(crate) fn new(input_shape: Vec<usize>, nodes: Vec<Node<F, S, PCS>>) -> Self {
        Self {
            input_shape,
            output_shape: nodes.last().unwrap().shape(),
            nodes,
            phantom: PhantomData,
        }
    }

    pub(crate) fn evaluate(&self, input: QArray<QSmallType>) -> QArray<QSmallType> {
        let mut output = input;
        for node in &self.nodes {
            println!(
                "****************** Evaluating {} ******************",
                node.type_name()
            );
            output = node.evaluate(output);
            println!("Output: {}", output);
        }
        output
    }

    pub(crate) fn padded_evaluate(&self, input: QArray<QSmallType>) -> QArray<QSmallType> {
        // TODO sanity check: input shape matches model input shape

        let mut output = input.compact_resize(
            // TODO this functionality is so common we might as well make it an #[inline] function
            self.input_shape
                .iter()
                .map(|x| x.next_power_of_two())
                .collect(),
            0,
        );

        for node in &self.nodes {
            output = node.padded_evaluate(output);
        }

        // TODO switch to reference in reshape?
        output.compact_resize(self.output_shape.clone(), 0)
    }
}
