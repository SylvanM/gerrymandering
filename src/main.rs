use itertools::Itertools;
use rand::Rng;

/**
 * Given 3*N students, assign them into groups so that the expected number of 
 * passing groups is maximized
 */
fn gerrymander(students: Vec<f64>) -> Vec<[f64 ; 3]> {
    let N = students.len() / 3;
    let mut sorted_students = students.clone();
    sorted_students.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let mut groups = vec![ [0.0 ; 3] ; students.len() / 3 ];

    for i in 0..sorted_students.len() {
        groups[i % N][i / N] = sorted_students[i];
    }

    groups
}

/**
 * Brute-force computes the best organization of groups
 */
fn bf_gerrymander(students: Vec<f64>) -> Vec<[f64 ; 3]> {
    let N = students.len() / 3;

    let mut max_grouping = vec![ [0.0 ; 3] ; students.len() / 3 ]; 
    let mut max_grouping_val = 0.0;

    for perm in students.iter().permutations(students.len()) {
        let mut groups = vec![ [0.0 ; 3] ; students.len() / 3 ];

        for i in 0..perm.len() {
            groups[i % N][i / N] = *perm[i];
        }

        let groups_val = passing_groups(groups.clone());

        if groups_val > max_grouping_val {
            max_grouping = groups;
            max_grouping_val = groups_val;
        }
    }

    max_grouping
}

/**
 * Given a partition of students, computes the expected number of groups 
 * that pass.
 */
fn passing_groups(groups: Vec<[f64 ; 3]>) -> f64 {
    groups.iter().map( |v|
        v[0]*v[1]*v[2] + (1.0 - v[0])*v[1]*v[2] + v[0]*(1.0 - v[1])*v[2] + v[0]*v[1]*(1.0 - v[2])
    ).sum()
}

#[test]
fn test_approx() {
    for N in 3..=100 {
        let mut students = vec![0.0 ; 3 * N];
        for i in 0..students.len() {
            students[i] = rand::thread_rng().gen_range(0.0..=1.0);
        }

        let approx_val = passing_groups(gerrymander(students.clone()));
        let best_val = passing_groups(bf_gerrymander(students));

        println!("{:?}", approx_val / best_val);
    }
}

fn main() {
    let students = vec![0.3, 0.8, 0.7, 0.7, 0.9, 0.4, 0.3, 0.7, 0.9];
    let groups = gerrymander(students.clone());

    let best_groups = bf_gerrymander(students);

    for g in groups.clone() {
        println!("{:?}", g);
    }
    println!("{}", passing_groups(groups));

    println!();

    for g in best_groups.clone() {
        println!("{:?}", g);
    }
    println!("{}", passing_groups(best_groups));
}
