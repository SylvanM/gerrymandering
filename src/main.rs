#![feature(generic_const_exprs)]

pub mod graph;
pub mod util;

use std::collections::HashSet;

use itertools::Itertools;
use graph::*;
use rand::{thread_rng, Rng};

/**
 * Given 3*N students, assign them into groups so that the expected number of 
 * passing groups is maximized
 */
fn gerrymander(students: Vec<f64>) -> Vec<[f64 ; 3]> {
    let n = students.len() / 3;
    let mut sorted_students = students.clone();
    sorted_students.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let mut groups = vec![ [0.0 ; 3] ; students.len() / 3 ];

    for i in 0..sorted_students.len() {
        groups[i % n][i / n] = sorted_students[i];
    }

    groups
}

/**
 * Brute-force computes the best organization of groups
 */
fn bf_gerrymander(students: Vec<f64>) -> Vec<[f64 ; 3]> {
    let n = students.len() / 3;

    let mut max_grouping = vec![ [0.0 ; 3] ; students.len() / 3 ]; 
    let mut max_grouping_val = 0.0;

    for perm in students.iter().permutations(students.len()) {
        let mut groups = vec![ [0.0 ; 3] ; students.len() / 3 ];

        for i in 0..perm.len() {
            groups[i % n][i / n] = *perm[i];
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
    for n in 3..=100 {
        let mut students = vec![0.0 ; 3 * n];
        for i in 0..students.len() {
            students[i] = rand::thread_rng().gen_range(0.0..=1.0);
        }

        let approx_val = passing_groups(gerrymander(students.clone()));
        let best_val = passing_groups(bf_gerrymander(students));

        println!("{:?}", approx_val / best_val);
    }
}

// fn main() {
//     let students = vec![0.3, 0.8, 0.7, 0.7, 0.9, 0.4, 0.3, 0.7, 0.9];
//     let groups = gerrymander(students.clone());

//     let best_groups = bf_gerrymander(students);

//     for g in groups.clone() {
//         println!("{:?}", g);
//     }
//     println!("{}", passing_groups(groups));

//     println!();

//     for g in best_groups.clone() {
//         println!("{:?}", g);
//     }
//     println!("{}", passing_groups(best_groups));
// }

// MARK: OUR Algorithm, let's see how it does!

fn n_choose_k(n: u64, k: u64) -> u64 {
    if k > n {
        return 0;
    }

    let k = if k > n - k { n - k } else { k }; // Use symmetry property
    let mut result = 1;

    for i in 1..=k {
        result = result * (n - i + 1) / i; // Multiply first, then divide
    }

    result
}

#[test]
fn print_table() {
    for n in 1..=100 {
        print!("{}, ", n_choose_k(3 * n, 3));
    }
}

const N: usize = 6; // yes, we'll have to change this by hand. :(
const NODE_COUNT_TABLE: [usize ; 100] = [1, 20, 84, 220, 455, 816, 1330, 2024, 2925, 4060, 5456, 7140, 9139, 11480, 14190, 17296, 20825, 24804, 29260, 34220, 39711, 45760, 52394, 59640, 67525, 76076, 85320, 95284, 105995, 117480, 129766, 142880, 156849, 171700, 187460, 204156, 221815, 240464, 260130, 280840, 302621, 325500, 349504, 374660, 400995, 428536, 457310, 487344, 518665, 551300, 585276, 620620, 657359, 695520, 735130, 776216, 818805, 862924, 908600, 955860, 1004731, 1055240, 1107414, 1161280, 1216865, 1274196, 1333300, 1394204, 1456935, 1521520, 1587986, 1656360, 1726669, 1798940, 1873200, 1949476, 2027795, 2108184, 2190670, 2275280, 2362041, 2450980, 2542124, 2635500, 2731135, 2829056, 2929290, 3031864, 3136805, 3244140, 3353896, 3466100, 3580779, 3697960, 3817670, 3939936, 4064785, 4192244, 4322340, 4455100];
const NODE_COUNT: usize = NODE_COUNT_TABLE[N - 1];

// Returns the weight of a set of nodes, given a mapping of nodes to groups
fn set_weight(is: Vec<usize>, group_weights: Vec<f64>) -> f64 {
    is.iter().map(|v| 
        group_weights[*v]
    ).sum()
}

// Returns the squared weight
fn squared_weight(is: &Vec<usize>, group_weights: &Vec<f64>) -> f64 {
    is.iter().map(|v| 
        group_weights[*v] * group_weights[*v]
    ).sum()
}

fn is_ind_set(graph: Graph<NODE_COUNT, i8>, is: Vec<usize>) -> bool {
    for v in is.clone() {
        for u in is.clone() {
            if graph.edge_exists((v, u)) {
                return false
            }
        }
    }
    return true;
}

// Returns Some(claw) if there exists a claw in the graph that improves the current ste
fn improving_claw(graph: Graph<NODE_COUNT, i8>, is: Vec<usize>, group_weights: Vec<f64>) -> Option<Vec<usize>> {
    for claw in graph.claws_leq(3) {
        let mut talons = vec![0 ; claw.len() - 1];
        for i in 0..talons.len() {
            talons[i] = claw[i + 1];
        }

        // Check if talons improves the graph.
        let current_squared_weight = squared_weight(&is, &group_weights);

        let other_set = util::set_union(util::set_diff(is.clone(), graph.relative_neighborhood(talons.clone(), is.clone())), talons.clone());
        let other_set_squared_weight = squared_weight(&other_set, &group_weights);

        if other_set_squared_weight > current_squared_weight {
            // First, let's assert that other_set is an independent set.
            assert!(is_ind_set(graph, other_set.clone()));
            return Some(other_set)
        }
    }

    None
}

fn berman_approx(p: Vec<f64>) -> Vec<[usize ; 3]> {
    // First, construct the graph
    let mut graph = Graph::<NODE_COUNT, i8>::disconnected();

    // We'll define a table so that we can easily pair groups with graph indices
    let mut possible_groups = vec![ [0 ; 3] ; NODE_COUNT ];

    let mut i = 0;
    for group in (0..p.len()).combinations(3) {
        for j in 0..3 {
            possible_groups[i][j] = group[j];
        }

        i += 1;
    }

    let group_weights = Vec::from_iter(possible_groups.iter().map(|g| 
        p[g[0]] * p[g[1]] * p[g[2]] + (1.0 - p[g[0]]) * p[g[1]] * p[g[2]] + p[g[0]] * (1.0 - p[g[1]]) * p[g[2]] + p[g[0]] * p[g[1]] * (1.0 - p[g[2]])
    ));
    
    assert_eq!(i, NODE_COUNT);

    // now assign edges!
    for g in possible_groups.clone() {
        // We want to iterate through every group that shares something in common with this group

        let g_set: HashSet<_> = g.iter().collect();
        let g_index = possible_groups.iter().clone().position(|&t| g == t).unwrap();

        for other_g in (0..p.len()).combinations(3) {
            if g.iter().all(|item| other_g.contains(item)) { continue; } // don't count ourselves!
            if other_g.iter().any(|item| g_set.contains(item)) {
                // Then we do our thing! Connect them!
                let other_index = possible_groups.clone().iter().position(|&t| other_g == t).unwrap();
                graph.set_edge_weight((g_index, other_index), 1);
                graph.set_edge_weight((other_index, g_index), 1);
            }
        }
    }

    // All edges should be connected now. Now, we define a weighting, and run Berman on it.

    let mut max_ind_set = Vec::new();

    while let Some(improving_set) = improving_claw(graph, max_ind_set.clone(), group_weights.clone()) {
        max_ind_set = improving_set;
    }

    Vec::from_iter(max_ind_set.clone().iter().map(|index| possible_groups[*index]))
}

pub fn grouping_score(groups: Vec<[usize ; 3]>, p: Vec<f64>) -> f64 {
    let probs: Vec<[f64 ; 3]> = groups.into_iter().map(|group| 
        group.map(|s| p[s])
    ).collect();

    passing_groups(probs)
}

fn main() {

    // We are testing for each N
    let mut students = vec![0.0 ; 3 * N];
    for i in 0..students.len() {
        students[i] = thread_rng().gen_range(0.0..=1.0);
    }

    let naive_score_groups = gerrymander(students.clone());
    let berman_groups = berman_approx(students.clone());
    // let max_groups = bf_gerrymander(students.clone());

    let naive_score = passing_groups(naive_score_groups);
    let berman_score = grouping_score(berman_groups, students);
    // let max_score = passing_groups(max_groups);

    println!("----[N={}]----", N);
    // println!("Max score: {}", max_score);
    println!("Naive score: {}", naive_score, );
    println!("Berman groups: {}", berman_score, );
    println!("Naive advantage: {}", naive_score / berman_score);
}

// fn main() {
//     let trials = 100;
//     for n in 1..=1000 {

//         println!("---[ n = {} ]--- (Naive only)", n);

//         let mut avg_expected = 0.0;

//         for _ in 1..=trials { // trials
            
//             let mut students = vec![0.0 ; 3 * n];
//             for i in 0..students.len() {
//                 students[i] = thread_rng().gen_range(0.0..=1.0);
//             }

//             let groups = gerrymander(students);
//             let score = passing_groups(groups);

//             let incremental = (score as f64) / (trials as f64);
//             avg_expected += incremental;

//         }

//         println!("Average winning groups: {}", avg_expected);
//         println!("Average performance: {}", avg_expected / (n as f64));
//     }
// }