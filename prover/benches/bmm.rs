use hcs_common::compatibility::get_model_output;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn tf_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("tf_inference");
    group.bench_with_input(
        BenchmarkId::new("fully connected layer", "mnist"),
        &(),
        |b, _| {
            b.iter(|| {
                get_model_output("QFullyConnectedLayer", None);
            });
        },
    );
}

criterion_group!(benches, tf_inference);
criterion_main!(benches);