use ark_std::marker::PhantomData;

use ark_std::{log2, rand::RngCore};

use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly::MultilinearExtension;
use ark_poly_commit::{LabeledPolynomial, PolynomialCommitment};
use hcs_common::{InferenceProof, Model};
use hcs_common::{NodeCommitment, NodeCommitmentState, Poly, QArray, QSmallType, QTypeArray};

use crate::NodeOpsProve;

pub struct ModelProver<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    phantom: PhantomData<(F, S, PCS)>,
}

impl<F, S, PCS> ModelProver<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    pub fn padded_evaluate(
        model: &Model<F, S, PCS>,
        input: QArray<QSmallType>,
    ) -> QArray<QSmallType> {
        let input = input;
        // TODO sanity check: input shape matches model input shape

        let input = input.compact_resize(
            // TODO this functionality is so common we might as well make it an #[inline] function
            model
                .input_shape
                .iter()
                .map(|x| x.next_power_of_two())
                .collect(),
            0,
        );

        let mut output = QTypeArray::S(input);

        for node in &model.nodes {
            output = node.padded_evaluate(&output);
        }

        // TODO switch to reference in reshape?
        match output {
            QTypeArray::S(o) => o.compact_resize(model.output_shape.clone(), 0),
            _ => panic!("Output QArray type should be QSmallType"),
        }
    }

    pub fn prove_inference(
        model: &Model<F, S, PCS>,
        ck: &PCS::CommitterKey,
        rng: Option<&mut dyn RngCore>,
        sponge: &mut S,
        node_coms: &Vec<NodeCommitment<F, S, PCS>>,
        node_com_states: &Vec<NodeCommitmentState<F, S, PCS>>,
        input: QArray<QSmallType>,
    ) -> InferenceProof<F, S, PCS> {
        // TODO Absorb public parameters into s (to be determined what exactly)

        let output = input.compact_resize(
            model
                .input_shape
                .iter()
                .map(|x| x.next_power_of_two())
                .collect(),
            0,
        );

        let output_f: Vec<F> = output.values().iter().map(|x| F::from(*x)).collect();

        let mut output = QTypeArray::S(output);

        // First pass: computing node values
        // TODO handling F and QSmallType is inelegant; we might want to switch
        // to F for IO in NodeOps::prove
        let mut node_outputs = vec![output.clone()];
        let mut node_output_mles = vec![Poly::from_evaluations_vec(
            log2(output_f.len()) as usize,
            output_f,
        )];

        for node in &model.nodes {
            output = node.padded_evaluate(&output);

            let output_f: Vec<F> = match &output {
                QTypeArray::S(o) => o.values().iter().map(|x| F::from(*x)).collect(),
                QTypeArray::L(o) => o.values().iter().map(|x| F::from(*x)).collect(),
            };

            node_outputs.push(output.clone());
            node_output_mles.push(Poly::from_evaluations_vec(
                log2(output_f.len()) as usize,
                output_f,
            ));
        }

        // Committing to node outputs as MLEs (individual per node for now)
        let labeled_output_mles: Vec<LabeledPolynomial<F, Poly<F>>> = node_output_mles
            .iter()
            .map(|mle|
            // TODO change dummy label once we e.g. have given numbers to the
            // nodes in the model: fc_1, fc_2, relu_1, etc.
            // TODO maybe we don't need to clone, if `prove` can take a reference
            LabeledPolynomial::new(
                "dummy".to_string(),
                mle.clone(),
                None,
                None,
            ))
            .collect();

        let (output_coms, output_com_states) = PCS::commit(ck, &labeled_output_mles, rng).unwrap();

        // Absorb all commitments into the sponge
        sponge.absorb(&output_coms);

        // TODO Prove that all commited NIOs live in the right range (to be
        // discussed)

        let mut node_proofs = Vec::new();

        // Second pass: proving
        for (((((node, node_com), node_com_state), values), l_v_coms), v_coms_states) in model
            .nodes
            .iter()
            .zip(node_coms.iter())
            .zip(node_com_states.iter())
            .zip(labeled_output_mles.windows(2))
            .zip(output_coms.windows(2))
            .zip(output_com_states.windows(2))
        {
            node_proofs.push(node.prove(
                ck,
                sponge,
                node_com,
                node_com_state,
                &values[0],
                &l_v_coms[0],
                &v_coms_states[0],
                &values[1],
                &l_v_coms[1],
                &v_coms_states[1],
            ));
        }

        // Opening model IO
        // TODO maybe this can be made more efficient by not committing to the
        // output nodes and instead working witht their plain values all along,
        // but that would require messy node-by-node handling
        let input_node = node_outputs.first().unwrap();
        let input_node_f = node_output_mles.first().unwrap().to_evaluations();
        let input_labeled_value = labeled_output_mles.first().unwrap();
        let input_node_com = output_coms.first().unwrap();
        let input_node_com_state = output_com_states.first().unwrap();

        let output_node = node_outputs.last().unwrap();
        let output_node_f = node_output_mles.last().unwrap().to_evaluations();
        let output_labeled_value = labeled_output_mles.last().unwrap();
        let output_node_com = output_coms.last().unwrap();
        let output_node_com_state = output_com_states.last().unwrap();

        // Absorb the model IO output and squeeze the challenge point
        // Absorb the plain output and squeeze the challenge point
        sponge.absorb(&input_node_f);
        sponge.absorb(&output_node_f);
        let input_challenge_point =
            sponge.squeeze_field_elements(log2(input_node_f.len()) as usize);
        let output_challenge_point =
            sponge.squeeze_field_elements(log2(output_node_f.len()) as usize);

        // TODO we have to pass rng, not None, but it has been moved before
        // fix this once we have decided how to handle the cumbersome
        // Option<&mut rng...>
        let input_opening_proof = PCS::open(
            ck,
            [input_labeled_value],
            [input_node_com],
            &input_challenge_point,
            sponge,
            [input_node_com_state],
            None,
        )
        .unwrap();

        // TODO we have to pass rng, not None, but it has been moved before
        // fix this once we have decided how to handle the cumbersome
        // Option<&mut rng...>
        let output_opening_proof = PCS::open(
            ck,
            [output_labeled_value],
            [output_node_com],
            &output_challenge_point,
            sponge,
            [output_node_com_state],
            None,
        )
        .unwrap();

        // TODO prove that inputs match input commitments?
        InferenceProof {
            inputs_outputs: vec![input_node.clone(), output_node.clone()],
            node_value_commitments: output_coms,
            node_proofs,
            opening_proofs: vec![input_opening_proof, output_opening_proof],
        }
    }

    pub fn commit(
        model: &Model<F, S, PCS>,
        ck: &PCS::CommitterKey,
        rng: Option<&mut dyn RngCore>,
    ) -> Vec<(NodeCommitment<F, S, PCS>, NodeCommitmentState<F, S, PCS>)> {
        let _rng = rng;
        // TODO blindly passing None, likely need to change to get hiding
        model.nodes.iter().map(|n| n.commit(ck, None)).collect()
    }
}
