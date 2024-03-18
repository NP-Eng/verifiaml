use ark_std::rand::RngCore;

use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use ark_poly_commit::{LabeledCommitment, LabeledPolynomial, PolynomialCommitment};

use crate::model::nodes::{NodeOpsCommon, NodeOpsNative};
use crate::{model::nodes::Node, quantization::QSmallType};

use self::qarray::QTypeArray;
use self::{nodes::NodeProof, qarray::QArray};

pub mod nodes;
pub mod qarray;

pub type Poly<F> = DenseMultilinearExtension<F>;
pub type LabeledPoly<F> = LabeledPolynomial<F, DenseMultilinearExtension<F>>;

pub struct InferenceProof<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    // Model input tensors in plain
    pub inputs: Vec<QTypeArray>,

    // Model output tensors in plain
    pub outputs: Vec<QTypeArray>,

    // Commitments to each of the node values
    pub node_value_commitments: Vec<LabeledCommitment<PCS::Commitment>>,

    // Proofs of evaluation of each of the model's nodes
    pub node_proofs: Vec<NodeProof<F, S, PCS>>,

    // Proofs of opening of each of the model's inputs
    pub input_opening_proofs: Vec<PCS::Proof>,

    // Proofs of opening of each of the model's outputs
    pub output_opening_proofs: Vec<PCS::Proof>,
}

// TODO change the functions that receive vectors to receive slices instead whenever it makes sense

// TODO: for now, we require all nodes to use the same PCS; this might change
// in the future
pub struct Model<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub nodes: Vec<Node<F, S, PCS>>,
}

impl<F, S, PCS> Model<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    pub fn new(input_shape: Vec<usize>, nodes: Vec<Node<F, S, PCS>>) -> Self {
        // An empty model would cause panics later down the line e.g. when
        // determining the number of variables needed to commit to it.
        assert!(!nodes.is_empty(), "A model cannot have no nodes",);

        Self {
            input_shape,
            output_shape: nodes.last().unwrap().shape(),
            nodes,
        }
    }

    pub fn input_shape(&self) -> &Vec<usize> {
        &self.input_shape
    }

    pub fn setup_keys<R: RngCore>(
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

    pub fn evaluate(&self, input: QArray<QSmallType>) -> QArray<QSmallType> {
        let mut output = QTypeArray::S(input);
        for node in &self.nodes {
            output = node.evaluate(&output);
        }

        match output {
            QTypeArray::S(o) => o,
            _ => panic!("Output QArray type should be QSmallType"),
        }
    }
}
