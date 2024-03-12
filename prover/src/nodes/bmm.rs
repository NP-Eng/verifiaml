use std::rc::Rc;

use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly::{MultilinearExtension, Polynomial};
use ark_poly_commit::{LabeledCommitment, LabeledPolynomial, PolynomialCommitment};
use ark_std::rand::RngCore;
use ark_sumcheck::ml_sumcheck::{protocol::ListOfProductsOfPolynomials, MLSumcheck};
use hcs_common::{
    BMMNode, BMMNodeCommitment, BMMNodeCommitmentState, BMMNodeProof, LabeledPoly, NodeCommitment,
    NodeCommitmentState, NodeOpsCommon, NodeProof, Poly, QArray, QLargeType, QTypeArray,
};

use crate::NodeOpsProve;

impl<F, S, PCS> NodeOpsProve<F, S, PCS> for BMMNode<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    // This function naively computes entries which are known to be zero. It is
    // meant to exactly mirror the proof-system multiplication proved by the
    // sumcheck argument. Requantisation and shifting are also applied to these
    // trivial entries, as the proof system does.
    fn padded_evaluate(&self, input: &QTypeArray) -> QTypeArray {
        let input = match input {
            QTypeArray::S(i) => i,
            _ => panic!("BMM node expects QSmallType as its QArray input type"),
        };

        let padded_dims = (1 << self.padded_dims_log.0, 1 << self.padded_dims_log.1);

        // Sanity checks
        // TODO systematise
        assert_eq!(
            input.num_dims(),
            1,
            "Incorrect shape: BMM node expects a 1-dimensional input array"
        );

        assert_eq!(
            padded_dims.0,
            input.len(),
            "Length mismatch: Padded fully connected node expected input with {} elements, got {} elements instead",
            padded_dims.0,
            input.len()
        );

        let input: QArray<QLargeType> = input.cast();

        // TODO this is a bigger question: can this overflow an i8? Supposedly the point of quantisation
        // is that input-by-weight products can be computed in i8. To be safe, let us use the large type here
        let shifted_input = input - self.input_zero_point as QLargeType;

        let mut accumulators = self.padded_bias.values().clone();

        // TODO this can be made more elegant (efficient?) using addition of QArrays after defining suitable operators

        // TODO since we have acumulators, this can be done more efficiently going row-wise to avoid re-caching the input
        for col in 0..padded_dims.1 {
            // TODO does the compiler realise it doesn't need to access accumulators[col] on every iteration of the inner loop? ow change
            for row in 0..padded_dims.0 {
                accumulators[col] += shifted_input[row]
                    * (self.padded_weights[row * padded_dims.1 + col] as QLargeType)
            }
        }

        let output = QArray::new(accumulators, vec![padded_dims.1]);

        QTypeArray::L(output)
    }

    fn prove(
        &self,
        ck: &PCS::CommitterKey,
        sponge: &mut S,
        node_com: &NodeCommitment<F, S, PCS>,
        node_com_state: &NodeCommitmentState<F, S, PCS>,
        input: &LabeledPoly<F>,
        input_com: &LabeledCommitment<PCS::Commitment>,
        input_com_state: &PCS::CommitmentState,
        output: &LabeledPoly<F>,
        output_com: &LabeledCommitment<PCS::Commitment>,
        output_com_state: &PCS::CommitmentState,
    ) -> NodeProof<F, S, PCS> {
        let (weight_com, bias_com) = match node_com {
            NodeCommitment::BMM(BMMNodeCommitment {
                weight_com,
                bias_com,
            }) => (weight_com, bias_com),
            _ => panic!("BMMNode::prove expected node commitment of type BMMNodeCommitment"),
        };

        let (weight_com_state, bias_com_state) = match node_com_state {
            NodeCommitmentState::BMM(BMMNodeCommitmentState {
                weight_com_state,
                bias_com_state,
            }) => (weight_com_state, bias_com_state),
            _ => panic!(
                "BMMNode::prove expected node commitment state of type BMMNodeCommitmentState"
            ),
        };

        // We can squeeze directly, since the sponge has already absorbed all the
        // commitments in Model::prove_inference
        let r: Vec<F> = sponge.squeeze_field_elements(self.padded_dims_log.1);

        let i_z_p_f = F::from(self.input_zero_point);

        // (f - zero-point)^
        let shifted_input_mle = Poly::from_evaluations_vec(
            input.num_vars(),
            input.polynomial().iter().map(|x| *x - i_z_p_f).collect(),
        );

        // TODO consider whether this can be done once and stored
        let weights_f = self
            .padded_weights
            .values()
            .iter()
            .map(|w| F::from(*w))
            .collect();

        // Dual of the MLE of the row-major flattening of the weight matrix
        let weight_mle = Poly::from_evaluations_vec(self.com_num_vars(), weights_f);

        // TODO consider whether this can be done once and stored
        let bias_f = self
            .padded_bias
            .values()
            .iter()
            .map(|w| F::from(*w))
            .collect();
        // Dual of the MLE of the bias vector
        let bias_mle = Poly::from_evaluations_vec(self.padded_dims_log.1, bias_f);

        let bias_opening_value = bias_mle.evaluate(&r);
        let output_opening_value = output.evaluate(&r);

        // Constructing the sumcheck polynomial
        // g(x) = (input - zero_point)^(x) * W^(r, x),
        let bound_weight_mle = weight_mle.fix_variables(&r);
        let mut g = ListOfProductsOfPolynomials::new(self.padded_dims_log.0);

        // TODO we are cloning the input here, can we do better?
        g.add_product(
            vec![shifted_input_mle, bound_weight_mle]
                .into_iter()
                .map(Rc::new)
                .collect::<Vec<_>>(),
            F::one(),
        );

        let (sumcheck_proof, prover_state) =
            MLSumcheck::<F, S>::prove_as_subprotocol(&g, sponge).unwrap();

        // The prover computes the claimed evaluations of weight_mle and
        // input_mle at the random challenge point
        // s := prover_state.randomness, the list of random values sampled by
        // the verifier during sumcheck. Note that this is different from r
        // above.
        //
        // We need to reveal g(s) by opening input^ at s and weight^ at s || r;
        // and also open output^ and bias^ at r
        let claimed_evaluations: Vec<F> = g
            .flattened_ml_extensions
            .iter()
            .map(|x| x.evaluate(&prover_state.randomness))
            .collect();

        // Recall that the first factor of g was the *shifted* dual input
        // (input - zero_point)^
        let input_opening_value = claimed_evaluations[0] + i_z_p_f;
        let weight_opening_value = claimed_evaluations[1];

        let input_opening_proof = PCS::open(
            ck,
            [input],
            [input_com],
            &prover_state.randomness,
            sponge,
            [input_com_state],
            None,
        )
        .unwrap();

        let weight_opening_proof = PCS::open(
            ck,
            [&LabeledPolynomial::new(
                "weight_mle".to_string(),
                weight_mle,
                Some(1),
                None,
            )],
            [weight_com],
            &r.clone()
                .into_iter()
                .chain(prover_state.randomness)
                .collect(),
            sponge,
            [weight_com_state],
            None,
        )
        .unwrap();

        // TODO: b and o are opened at the same point, so they could be opened
        // with a single call to PCS::open
        let output_bias_opening_proof = PCS::open(
            ck,
            [
                output,
                &LabeledPolynomial::new("bias_mle".to_string(), bias_mle, Some(1), None),
            ],
            [output_com, bias_com],
            &r,
            sponge,
            [output_com_state, bias_com_state],
            None,
        )
        .unwrap();

        NodeProof::BMM(BMMNodeProof {
            sumcheck_proof,
            input_opening_proof,
            input_opening_value,
            weight_opening_proof,
            weight_opening_value,
            output_bias_opening_proof,
            output_opening_value,
            bias_opening_value,
        })
    }

    fn commit(
        &self,
        ck: &PCS::CommitterKey,
        rng: Option<&mut dyn RngCore>,
    ) -> (NodeCommitment<F, S, PCS>, NodeCommitmentState<F, S, PCS>) {
        // TODO should we separate the associated commitment type into one with state and one without?
        let padded_weights_f: Vec<F> = self
            .padded_weights
            .values()
            .iter()
            .map(|w| F::from(*w))
            .collect();

        // TODO part of this code is duplicated in prove, another hint that this should probs
        // be stored
        let weight_poly = LabeledPolynomial::new(
            "weight_poly".to_string(),
            Poly::from_evaluations_vec(self.com_num_vars(), padded_weights_f),
            Some(1),
            None,
        );

        let padded_bias_f: Vec<F> = self
            .padded_bias
            .values()
            .iter()
            .map(|b| F::from(*b))
            .collect();

        let bias_poly = LabeledPolynomial::new(
            "bias_poly".to_string(),
            Poly::from_evaluations_vec(self.padded_dims_log.1, padded_bias_f),
            Some(1),
            None,
        );

        let coms = PCS::commit(ck, vec![&weight_poly, &bias_poly], rng).unwrap();

        (
            NodeCommitment::BMM(BMMNodeCommitment {
                weight_com: coms.0[0].clone(),
                bias_com: coms.0[1].clone(),
            }),
            NodeCommitmentState::BMM(BMMNodeCommitmentState {
                weight_com_state: coms.1[0].clone(),
                bias_com_state: coms.1[1].clone(),
            }),
        )
    }
}
