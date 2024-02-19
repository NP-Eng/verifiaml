use ark_std::{log2, rand::RngCore};

use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use ark_poly_commit::{LabeledPolynomial, PolynomialCommitment};

use crate::model::nodes::{NodeOps, NodeOpsSNARK};
use crate::{model::nodes::Node, quantization::QSmallType};

use self::{
    nodes::{NodeCommitment, NodeCommitmentState, NodeProof},
    qarray::QArray,
};

mod examples;
mod nodes;
mod qarray;
mod reshaping;

pub(crate) type Poly<F> = DenseMultilinearExtension<F>;

pub(crate) struct InferenceProof<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    // Model output tensors
    outputs: Vec<QArray<QSmallType>>,

    // Proofs of evaluation of each of the model's nodes
    node_proofs: Vec<NodeProof>,

    // Proofs of opening of each of the model's outputs
    opening_proofs: PCS::Proof,
}

// TODO change the functions that receive vectors to receive slices instead whenever it makes sense

// TODO: for now, we require all nodes to use the same PCS; this might change
// in the future
pub(crate) struct Model<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    nodes: Vec<Node<F, S, PCS>>,
}

impl<F, S, PCS> Model<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    pub(crate) fn new(input_shape: Vec<usize>, nodes: Vec<Node<F, S, PCS>>) -> Self {
        // An empty model would cause panics later down the line e.g. when
        // determining the number of variables needed to commit to it.
        assert!(!nodes.is_empty(), "A model cannot have no nodes",);

        Self {
            input_shape,
            output_shape: nodes.last().unwrap().shape(),
            nodes,
        }
    }

    pub(crate) fn setup_keys<R: RngCore>(
        &self,
        rng: &mut R,
    ) -> Result<(PCS::CommitterKey, PCS::VerifierKey), PCS::Error> {
        let num_vars = self.nodes.iter().map(|n| n.com_num_vars()).max().unwrap();

        let pp = PCS::setup(1, Some(num_vars), rng).unwrap();

        // Make sure supported_degree, supported_hiding_bound and
        // enforced_degree_bounds have a consistent meaning across ML PCSs and
        // we are using them securely.
        PCS::trim(&pp, 0, 0, None)
    }

    pub(crate) fn evaluate(&self, input: QArray<QSmallType>) -> QArray<QSmallType> {
        let mut output = input;
        for node in &self.nodes {
            output = node.evaluate(output);
        }
        output
    }

    /// Unlike the node's `padded_evaluate`, the model's `padded_evaluate` accepts unpadded input
    /// and first re-sizes it before running inference.
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

    pub(crate) fn prove_inference(
        &self,
        ck: &PCS::CommitterKey,
        rng: Option<&mut dyn RngCore>,
        s: &mut S,
        node_commitments: Vec<NodeCommitment<F, S, PCS>>,
        input: QArray<QSmallType>,
    ) -> InferenceProof<F, S, PCS> {
        unimplemented!();
    }

    pub(crate) fn commit(
        &self,
        ck: &PCS::CommitterKey,
        rng: Option<&mut dyn RngCore>,
    ) -> Vec<(NodeCommitment<F, S, PCS>, NodeCommitmentState<F, S, PCS>)> {
        // TODO blindly passing None, likely need to change to get hiding
        self.nodes.iter().map(|n| n.commit(ck, None)).collect()
    }
}
