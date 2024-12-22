use matrix::Matrix;
use matrix_kit::*;
use algebra_kit::algebra::*;

use itertools::Itertools;


fn bp_to_set<const N: usize>(bits: usize) -> Vec<usize> {
    let mut set = Vec::<usize>::new();
    for v in 0..N {
        // check if v is in this vertex selection
        if bits & (1 << v) != 0 {
            set.push(v);
        }
    }
    set
}

fn set_to_bp(set: Vec<usize>) -> usize {
    let mut bit_pattern = 0;
    for v in set {
        bit_pattern |= 1 << v;
    }
    bit_pattern
}

pub trait WeightType: Ring + PartialOrd + Copy { /* Just need to group these traits :) */ }
impl<T> WeightType for T where T: Ring + PartialOrd + Copy {}

/// A directed graph with N nodes, with edge weights of type W
#[derive(Clone, Copy)]
pub struct Graph<const N: usize, W: WeightType> where [() ; N * N]: Sized {
    pub adjacencies: Matrix<N, N, W>
}

pub type Edge = (usize, usize);
pub type Path = Vec<Edge>;

impl<const N: usize, W: WeightType> Graph<N, W> where [(); N * N]: {

    // MARK: Constructors

	/// Creates a graph with a certain adjacency matrix
	pub fn from_adj_matrix(mat: Matrix<N, N, W>) -> Graph<N, W> {
		Graph { adjacencies: mat }
	}

	/// Creates a graph with N nodes and no edges
	pub fn disconnected() -> Graph<N, W> {
		Graph { adjacencies: Matrix::<N, N, W>::new() }
	}

    /// Creates an undirected graph from a list of adjacencies
    pub fn from_undir_adjacency_list(adj_list: Vec<Edge>) -> Graph<N, W> {
        let mut adj_mat = Matrix::<N, N, W>::new();
        for (u, v) in adj_list {
            adj_mat[u][v] = W::one();
            adj_mat[v][u] = W::one();
        }
        Graph { adjacencies: adj_mat }
    }

    /// Creates a directed graph from a list of adjacencies
    pub fn from_dir_adjacency_list(adj_list: Vec<Edge>) -> Graph<N, W> {
        let mut adj_mat = Matrix::<N, N, W>::new();
        for (u, v) in adj_list {
            adj_mat[u][v] = W::one();
        }
        Graph { adjacencies: adj_mat }
    }

    // MARK: Basic Properties

    /// Returns the weight of an edge e = (u, v)
    pub fn get_edge_weight(self, e: Edge) -> W {
        let (u, v) = e;
        self.adjacencies[u][v]
    }

    /// Sets the edge weight of an edge (u, v)
    pub fn set_edge_weight(&mut self, e: Edge, weight: W) {
        let (u, v) = e;
        self.adjacencies[u][v] = weight
    }

    /// Returns whether or not there is a directed edge (u, v)
    pub fn edge_exists(self, e: Edge) -> bool {
        self.get_edge_weight(e) != W::zero()
    }

    /// Returns a set of the directed neighbors of the vertex v
    /// 
    /// O(n)
    pub fn get_neighbors(self, v: usize) -> Vec<usize> {
        let mut neighbors = Vec::new();

        for u in 0..N {
            if self.edge_exists((u, v)) {
                neighbors.push(u);
            }
        }

        neighbors
    }

    /// Returns a list of all the edges in this graph 
    pub fn get_edges(self) -> Vec<Edge> {
        let mut edges = Vec::new();

        for u in 0..N {
            for v in 0..N {
                if self.edge_exists((u, v)) {
                    edges.push((u, v));
                }
            }
        }

        edges
    }

    /// Returns a list of all undirected edges in this graph 
    pub fn get_undirected_edges(self) -> Vec<Edge> {
        let mut edges = Vec::new();

        for u in 0..N {
            for v in 0..N {
                if self.edge_exists((u, v)) || self.edge_exists((u, v)) {
                    // add this edge, if we haven't already
                    if edges.contains(&(v, u)) {
                        continue;
                    }
                    edges.push((u, v));
                }
            }
        }

        edges
    }

    /// Returns TRUE if e1 is toucing e2, in a DIRECTED manor
    pub fn edges_adjacent_directed(e1: Edge, e2: Edge) -> bool {
        let (_, v_last) = e1;
        let (u_first, _) = e2;
        v_last == u_first
    }

    /// Returns TRUE if e1 and e2 share a vertex. FALSE if they are the same edge or do not share a vertex.
    pub fn edges_adjacent_undirected(e1: Edge, e2: Edge) -> bool {
        if e1 == e2 {
            false
        } else {
            let (u1, v1) = e1;
            let (u2, v2) = e2;

            u1 == u2 || u1 == v2 || v1 == u2 || v1 == v2
        }
    }

    // MARK: Paths

    /// Returns the set of all paths from s to t, as a list of paths
    pub fn all_directed_paths(self, s: usize, t: usize) -> Vec<Path> {
        // This is exponential anyway, why not write a brute force algorithm for it?
        // Generate ALL POSSIBLE edge sets, and check if it's a path from u to v

        let mut paths = Vec::new();

        let edges = self.get_edges(); // We'll just subsets as binary strings of this length
        
        for edge_set_binary in 1..(1 << edges.len()) { // We won't start with the empty set
            let mut edge_set = Vec::new();
            for i in 0..edges.len() {
                if edge_set_binary & (1 << i) != 0 {
                    edge_set.push(edges[i]);
                }
            }

            // Now check if this edge set is a path!

            // Does it start with s and end with t?
            if edge_set[0].0 != s || edge_set[edge_set.len() - 1].1 != t {
                continue;
            }

            // Are all the edges in the path adjacent?
            let mut all_adjacent = true;
            for i in 1..edge_set.len() {
                if !Self::edges_adjacent_directed(edge_set[i - 1], edge_set[i]) {
                    all_adjacent = false;
                    break;
                }
            }
            if !all_adjacent {
                continue;
            }

            paths.push(edge_set);
        }

        paths
    }

    /// Returns the set of all undirected paths from s to t, as a list of paths
    pub fn all_undirected_paths(self, s: usize, t: usize) -> Vec<Path> {
        // This is exponential anyway, why not write a brute force algorithm for it?
        // Generate ALL POSSIBLE edge sets, and check if it's a path from u to v

        let mut paths = Vec::new();

        let edges = self.get_undirected_edges(); // We'll just subsets as binary strings of this length
        
        for edge_set_binary in 1..(1 << edges.len()) { // We won't start with the empty set
            let mut edge_set = Vec::new();
            for i in 0..edges.len() {
                if edge_set_binary & (1 << i) != 0 {
                    edge_set.push(edges[i]);
                }
            }

            // Now check if this edge set is a path!

            // Does it start with s and end with t?
            if !((edge_set[0].0 == s || edge_set[0].1 == s) && (edge_set[edge_set.len() - 1].0 == t || edge_set[edge_set.len() - 1].1 == t)) {
                continue;
            }

            // Are s and t only included ONCE?
            // This is NOT a good way of doing this, since it rules out cycles on the start and end, which SHOULD be allowed
            // #[warn(unsafe_code)]
            // let mut s_occurrences = 0;
            // for e in edge_set.clone() {
            //     let (u, v) = e;
            //     if u == s {
            //         s_occurrences += 1;
            //     }
            //     if v == s {
            //         s_occurrences += 1;
            //     }
            // }
            // let mut t_occurrences = 0;
            // for e in edge_set.clone() {
            //     let (u, v) = e;
            //     if u == t {
            //         t_occurrences += 1;
            //     }
            //     if v == t {
            //         t_occurrences += 1;
            //     }
            // }
            // if !(s_occurrences == 1 && t_occurrences == 1) {
            //     continue;
            // }

            // Are all the edges in the path adjacent? We can't count on edges being "in order"
            let mut all_adjacent = true;
            for i in 1..edge_set.len() {
                if !Self::edges_adjacent_undirected(edge_set[i - 1], edge_set[i]) {
                    all_adjacent = false;
                    break;
                }
            }
            if !all_adjacent {
                continue;
            }

            // Now, try to create a path out of each edge?
            let mut unsorted_vertices = Vec::<usize>::new();

            let mut desired_node = s;
            while true { // 

            }

            paths.push(edge_set);
        }

        paths
    }

	// MARK: Structures

	/**
	 * Given two subsets U and W of V (the vertices), returns a neighborhood of 
	 * U in W
	 */
	pub fn relative_neighborhood(self, u_vec: Vec<usize>, w_vec: Vec<usize>) -> Vec<usize> {
		let mut neighborhood = Vec::new();

		for w in w_vec {
			if u_vec.clone().iter().any(|&t| t == w) { // if w is in u already, cool!
				neighborhood.push(w);
				continue;
			}

			for u in u_vec.clone() { // check if any u maps into w
				if self.edge_exists((u, w)) {
					neighborhood.push(w);
					break;
				}
			}
		}

		neighborhood
	}

	/**
	 * Finds all d-claws. The return value is a vector of sets of nodes. In each set, 
	 * the first element is the "center" of the claw and the rest are the talons.
	 */
	pub fn claws(self, d: usize) -> Vec<Vec<usize>> {
		let mut claws = Vec::new();

		for node in 0..N {
			// Check all neighbors of this node to see if we have a claw.

			for d_cluster_ptrs in self.get_neighbors(node).iter().combinations(d) {
				// check if d_cluster forms a claw by ensuring that no
				// two of the nodes are attached to each other.
				let d_cluster = d_cluster_ptrs.iter().map(|&p| p);

				// println!("Analyzing talon: {:?}", d_cluster);

				let mut is_claw = true;
				for node_pair in d_cluster.clone().combinations(2) {
					if self.edge_exists((*node_pair[0], *node_pair[1])) {
						is_claw = false;
						break;
					}
				}

				if is_claw {

					let mut claw = vec![node];
					for d in d_cluster {
						claw.push(*d);
					}
					
					claws.push(claw);
				}
			}
		}

		claws
	}

	// Returns all claws of size <= d
	pub fn claws_leq(self, d: usize) -> Vec<Vec<usize>> {
		let mut claws = Vec::new();

		for k in 1..=d {
			claws.extend(self.claws(k));
		}

		claws
	}
 
    // MARK: Transforms

    /// Computes the complement of this graph
    pub fn complement(self) -> Graph<N, W> {
        let all_ones = Matrix::<N, N, W>::from_flatmap([W::one() ; N * N]);
        let complememt_adj_mat = all_ones - self.adjacencies;
        Graph { adjacencies: complememt_adj_mat }
    }

    // MARK: Vertex Sets

    /// Inefficiently computes all cliques in a graph,
    /// returning a set of sets
    pub fn cliques(self) -> Vec<Vec<usize>> {
        // Iterate through every possible subset of vertices, and check if its a clique. If it is,
        // add that subset to our list!
        let mut cliques = Vec::<Vec<usize>>::new();

        for subset_bitmap in 1..(1 << N) { // we start at 1 because we don't care about 0 which represents choosing no vertices
            let selected_vertices = bp_to_set::<N>(subset_bitmap);

            // Now, check that every selected vertex is adjacent to every other selected vertex.
            let mut is_clique = true;
            
            for v in selected_vertices.clone() { // Check vertex v
                for u in selected_vertices.clone() { 
                    if v == u { continue; } // We don't care about self-adjacency

                    // Check that v is adjacent to u
                    if self.adjacencies[v][u] == W::zero() {
                        // this is not a clique!
                        is_clique = false;
                    }
                }
            }

            if is_clique {
                cliques.push(selected_vertices);
            }
        }

        cliques
    }

    /// Finds all independent sets in a graph
    pub fn independent_sets(self) -> Vec<Vec<usize>> {
        self.complement().cliques()
    }

    /// Finds all independent sets that contain a specific vertex
    pub fn independent_sets_containing(self, v: usize) -> Vec<Vec<usize>> {
        self.independent_sets()
            .into_iter()
            .filter(|is| is.contains(&v))
            .collect()
    }
	
}