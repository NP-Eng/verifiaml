use super::*;

#[test]
fn test_flatten_index_trivial() {
    let q = QArray::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    for i in 0..9 {
        assert_eq!(q.flatten_index(vec![i]), i);
    }
}

#[test]
fn test_flatten_index_2d() {
    let shape = vec![3, 3];
    let flattened: Vec<i32> = (1..=9).collect();
    let q = QArray::new(flattened, shape);
    assert_eq!(q.flatten_index(vec![0, 0]), 0);
    assert_eq!(q.flatten_index(vec![0, 1]), 1);
    assert_eq!(q.flatten_index(vec![0, 2]), 2);
    assert_eq!(q.flatten_index(vec![1, 0]), 3);
    assert_eq!(q.flatten_index(vec![1, 2]), 5);
    assert_eq!(q.flatten_index(vec![2, 0]), 6);
    assert_eq!(q.flatten_index(vec![2, 2]), 8);
}

#[test]
fn test_flatten_index_3d() {
    let shape = vec![2, 3, 4];
    let flattened: Vec<i32> = (1..=24).collect();
    let q = QArray::new(flattened, shape);
    assert_eq!(q.flatten_index(vec![0, 0, 0]), 0);
    assert_eq!(q.flatten_index(vec![0, 0, 1]), 1);
    assert_eq!(q.flatten_index(vec![0, 0, 2]), 2);
    assert_eq!(q.flatten_index(vec![0, 1, 0]), 4);
    assert_eq!(q.flatten_index(vec![1, 0, 0]), 12);
    assert_eq!(q.flatten_index(vec![1, 2, 3]), 23);
}

#[test]
fn test_resize_1d_pad() {
    let shape = vec![5];
    let flattened: Vec<i32> = (1..=5).collect();
    let qarray = QArray::new(flattened, shape);

    let padded = qarray.compact_resize(vec![7], 0);

    assert_eq!(padded.shape, vec![7]);
    assert_eq!(padded.flattened, vec![1, 2, 3, 4, 5, 0, 0]);
}

#[test]
fn test_resize_2d_pad() {
    let shape = vec![2, 3];
    let flattened: Vec<i32> = (1..=6).collect();
    let qarray = QArray::new(flattened, shape);

    let padded = qarray.compact_resize(vec![5, 4], 0);

    let expected = vec![1, 2, 3, 0, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    assert_eq!(padded.shape, vec![5, 4]);
    assert_eq!(padded.flattened, expected);
}

#[test]
fn test_resize_3d_pad() {
    let shape = vec![2, 3, 4];
    let flattened: Vec<i32> = (1..=24).collect();
    let qarray = QArray::new(flattened, shape);

    let padded = qarray.compact_resize(vec![3, 5, 7], 0);

    let expected = vec![
        1, 2, 3, 4, 0, 0, 0, 5, 6, 7, 8, 0, 0, 0, 9, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 13, 14, 15, 16, 0, 0, 0, 17, 18, 19, 20, 0, 0, 0, 21, 22, 23, 24, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];

    assert_eq!(padded.shape, vec![3, 5, 7]);
    assert_eq!(padded.flattened, expected);
}

#[test]
fn test_resize_1d_truncate() {
    let shape = vec![7];
    let original = vec![1, 2, 3, 4, 5, 0, 0];
    let qarray = QArray::new(original, shape);

    let expected: Vec<i32> = (1..=5).collect();

    let padded = qarray.compact_resize(vec![5], 0);

    assert_eq!(padded.shape, vec![5]);
    assert_eq!(padded.flattened, expected);
}

#[test]
fn test_resize_2d_truncate() {
    let shape = vec![5, 4];
    let original = vec![1, 2, 3, 0, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let qarray = QArray::new(original, shape);

    let padded = qarray.compact_resize(vec![2, 3], 0);

    let expected: Vec<i32> = (1..=6).collect();

    assert_eq!(padded.shape, vec![2, 3]);
    assert_eq!(padded.flattened, expected);
}

#[test]
fn test_resize_3d_truncate() {
    let shape = vec![3, 5, 7];
    let original = vec![
        1, 2, 3, 4, 0, 0, 0, 5, 6, 7, 8, 0, 0, 0, 9, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 13, 14, 15, 16, 0, 0, 0, 17, 18, 19, 20, 0, 0, 0, 21, 22, 23, 24, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    let qarray = QArray::new(original, shape);

    let padded = qarray.compact_resize(vec![2, 3, 4], 0);

    let expected: Vec<i32> = (1..=24).collect();

    assert_eq!(padded.shape, vec![2, 3, 4]);
    assert_eq!(padded.flattened, expected);
}

#[test]
fn test_resize_3d_mixed() {
    let shape = vec![2, 2, 3];
    let flattened: Vec<i32> = (1..=12).collect();
    let qarray = QArray::new(flattened, shape);

    let padded = qarray.compact_resize(vec![3, 1, 5], 0);

    let expected = vec![1, 2, 3, 0, 0, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0];

    assert_eq!(padded.shape, vec![3, 1, 5]);
    assert_eq!(padded.flattened, expected);
}
