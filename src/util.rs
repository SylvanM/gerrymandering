

/**
 * Computes the set difference of two sets
 */
pub fn set_diff<T: PartialEq + Copy>(a: Vec<T>, b: Vec<T>) -> Vec<T> {

	a.into_iter().filter(|item| !b.contains(item)).collect()

	// Vec::from_iter( a.iter().map(|&t| t).filter(|&x| b.iter().any(|&v| v == x)))

	// new_items.into_iter().filter(|item| !previous_items.contains(item)).collect();
}

/**
 * Computes the union of two sets
 */
pub fn set_union<T: PartialEq + Copy>(a: Vec<T>, b: Vec<T>) -> Vec<T> {
	let mut union_vec = a.clone();

	for x in b {
		if !a.iter().any(|&t| t == x) {
			union_vec.push(x);
		}
	}

	union_vec
}
