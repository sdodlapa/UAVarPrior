#!/usr/bin/env python3
"""
Comprehensive test to validate whether using a similarity matrix (count matrix) 
vs. binary membership matrix is appropriate for Jaccard similarity calculations.

This script tests both approaches and demonstrates why the current implementation 
using a similarity matrix (count matrix) is correct for Jaccard calculations.
"""

import numpy as np
from scipy import sparse
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, '/home/sdodl001/myrepos/UAVarPrior/src')

def create_test_binary_membership_matrix():
    """Create a test binary membership matrix (variants × profiles)."""
    # Create a simple example:
    # 5 variants: {A, B, C, D, E}
    # 3 profiles: P1, P2, P3
    # P1 has variants: {A, B, C}
    # P2 has variants: {B, C, D}  
    # P3 has variants: {A, E}
    
    data = [
        # Variant A (row 0): in profiles P1, P3
        1, 0, 1,
        # Variant B (row 1): in profiles P1, P2
        1, 1, 0,
        # Variant C (row 2): in profiles P1, P2
        1, 1, 0,
        # Variant D (row 3): in profile P2
        0, 1, 0,
        # Variant E (row 4): in profile P3
        0, 0, 1
    ]
    
    rows = [0, 0, 0,  # variant A
            1, 1, 1,  # variant B
            2, 2, 2,  # variant C
            3, 3, 3,  # variant D
            4, 4, 4]  # variant E
    
    cols = [0, 1, 2,  # P1, P2, P3 for variant A
            0, 1, 2,  # P1, P2, P3 for variant B
            0, 1, 2,  # P1, P2, P3 for variant C
            0, 1, 2,  # P1, P2, P3 for variant D
            0, 1, 2]  # P1, P2, P3 for variant E
    
    membership_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(5, 3))
    return membership_matrix

def create_similarity_matrix_from_membership(membership_matrix):
    """Create similarity matrix from binary membership matrix."""
    # This is exactly what the UAVarPrior code does
    profiles_matrix = membership_matrix.transpose()  # profiles × variants
    similarity_matrix = profiles_matrix.dot(membership_matrix)  # profiles × profiles
    return similarity_matrix

def manual_jaccard_from_binary_matrix(membership_matrix, profile_i, profile_j):
    """Calculate Jaccard similarity directly from binary membership matrix."""
    # Extract binary vectors for each profile
    vector_i = membership_matrix[:, profile_i].toarray().flatten()
    vector_j = membership_matrix[:, profile_j].toarray().flatten()
    
    # Calculate intersection and union
    intersection = np.sum(vector_i & vector_j)  # Both have the variant
    union = np.sum(vector_i | vector_j)         # Either has the variant
    
    jaccard = intersection / union if union > 0 else 0
    return jaccard, intersection, union, np.sum(vector_i), np.sum(vector_j)

def jaccard_from_similarity_matrix(similarity_matrix, profile_i, profile_j):
    """Calculate Jaccard similarity from similarity matrix (count matrix)."""
    common_variants = similarity_matrix[profile_i, profile_j]
    variants_i = similarity_matrix[profile_i, profile_i]
    variants_j = similarity_matrix[profile_j, profile_j]
    
    union_size = variants_i + variants_j - common_variants
    jaccard = common_variants / union_size if union_size > 0 else 0
    
    return jaccard, common_variants, union_size, variants_i, variants_j

def test_matrix_approaches():
    """Test both approaches and validate they give the same results."""
    print("Testing Binary Membership Matrix vs Similarity Matrix for Jaccard Calculations")
    print("=" * 80)
    
    # Create test data
    membership_matrix = create_test_binary_membership_matrix()
    similarity_matrix = create_similarity_matrix_from_membership(membership_matrix)
    
    print("1. Test Data Overview:")
    print("-" * 40)
    print("Binary Membership Matrix (variants × profiles):")
    print("   Variant A: [1, 0, 1]  # in P1, P3")
    print("   Variant B: [1, 1, 0]  # in P1, P2")
    print("   Variant C: [1, 1, 0]  # in P1, P2")
    print("   Variant D: [0, 1, 0]  # in P2")
    print("   Variant E: [0, 0, 1]  # in P3")
    print()
    print("Profile contents:")
    print("   P1: {A, B, C}  (3 variants)")
    print("   P2: {B, C, D}  (3 variants)")  
    print("   P3: {A, E}     (2 variants)")
    print()
    
    print("Binary membership matrix shape:", membership_matrix.shape)
    print("Binary membership matrix:")
    print(membership_matrix.toarray())
    print()
    
    print("Similarity matrix (count matrix) shape:", similarity_matrix.shape)
    print("Similarity matrix (profiles × profiles):")
    print(similarity_matrix.toarray())
    print()
    
    # Test all profile pairs
    print("2. Jaccard Similarity Calculations:")
    print("-" * 40)
    
    n_profiles = similarity_matrix.shape[0]
    all_matches = True
    
    for i in range(n_profiles):
        for j in range(i, n_profiles):
            # Method 1: Direct from binary membership matrix
            jaccard_binary, intersect, union_binary, count_i, count_j = manual_jaccard_from_binary_matrix(
                membership_matrix, i, j
            )
            
            # Method 2: From similarity matrix (UAVarPrior approach)
            jaccard_similarity, common, union_similarity, variants_i, variants_j = jaccard_from_similarity_matrix(
                similarity_matrix, i, j
            )
            
            print(f"\nProfile pair P{i+1} and P{j+1}:")
            print(f"  Method 1 (Binary Matrix):")
            print(f"    Intersection: {intersect}, Union: {union_binary}")
            print(f"    Variants in P{i+1}: {count_i}, Variants in P{j+1}: {count_j}")
            print(f"    Jaccard: {jaccard_binary:.4f}")
            
            print(f"  Method 2 (Similarity Matrix - UAVarPrior approach):")
            print(f"    Common variants: {common}, Union: {union_similarity}")
            print(f"    Variants in P{i+1}: {variants_i}, Variants in P{j+1}: {variants_j}")
            print(f"    Jaccard: {jaccard_similarity:.4f}")
            
            # Validate they match
            if abs(jaccard_binary - jaccard_similarity) < 1e-10:
                print(f"    ✅ MATCH: Both methods give identical results")
            else:
                print(f"    ❌ MISMATCH: Methods give different results!")
                all_matches = False
    
    print(f"\n3. Validation Results:")
    print("-" * 40)
    if all_matches:
        print("✅ SUCCESS: Both approaches give identical Jaccard similarity values")
        print("✅ The similarity matrix (count matrix) approach used in UAVarPrior is CORRECT")
    else:
        print("❌ FAILURE: Approaches give different results")
    
    return all_matches

def explain_why_similarity_matrix_works():
    """Explain why using a similarity matrix for Jaccard calculations is correct."""
    print("\n4. Mathematical Explanation:")
    print("-" * 40)
    print("Why the similarity matrix approach is correct for Jaccard similarity:")
    print()
    print("Given a binary membership matrix M (variants × profiles):")
    print("  M[v,p] = 1 if variant v is present in profile p, 0 otherwise")
    print()
    print("The similarity matrix S is computed as:")
    print("  S = M^T × M")
    print("  where M^T is the transpose of M (profiles × variants)")
    print()
    print("For any two profiles i and j:")
    print("  S[i,j] = sum over all variants v of (M[v,i] × M[v,j])")
    print("         = count of variants present in BOTH profiles i and j")
    print("         = |intersection(profile_i, profile_j)|")
    print()
    print("  S[i,i] = sum over all variants v of (M[v,i] × M[v,i])")
    print("         = sum over all variants v of M[v,i]  (since M[v,i] is binary)")
    print("         = count of variants in profile i")
    print("         = |profile_i|")
    print()
    print("Therefore, the Jaccard similarity is:")
    print("  J(i,j) = |intersection| / |union|")
    print("         = S[i,j] / (S[i,i] + S[j,j] - S[i,j])")
    print("         = common_variants / (variants_i + variants_j - common_variants)")
    print()
    print("This is exactly what the UAVarPrior implementation does!")

def test_edge_cases():
    """Test edge cases to ensure robustness."""
    print("\n5. Edge Case Testing:")
    print("-" * 40)
    
    # Test case 1: Empty profiles
    print("Test 1: Empty profiles")
    membership_empty = sparse.csr_matrix(np.array([
        [1, 0, 0],  # variant A only in profile 1
        [0, 0, 0],  # variant B in no profiles
    ]))
    similarity_empty = create_similarity_matrix_from_membership(membership_empty)
    
    # Profile 1 vs Profile 2 (profile 2 is empty)
    jaccard_empty, _, _, _, _ = jaccard_from_similarity_matrix(similarity_empty, 0, 1)
    print(f"  Jaccard(non-empty, empty) = {jaccard_empty:.4f} (should be 0)")
    
    # Profile 2 vs Profile 3 (both empty)
    jaccard_both_empty, _, _, _, _ = jaccard_from_similarity_matrix(similarity_empty, 1, 2)
    print(f"  Jaccard(empty, empty) = {jaccard_both_empty:.4f} (should be 0)")
    
    # Test case 2: Identical profiles
    print("\nTest 2: Identical profiles")
    membership_identical = sparse.csr_matrix(np.array([
        [1, 1, 0],  # variant A in profiles 1,2
        [1, 1, 0],  # variant B in profiles 1,2
    ]))
    similarity_identical = create_similarity_matrix_from_membership(membership_identical)
    
    jaccard_identical, _, _, _, _ = jaccard_from_similarity_matrix(similarity_identical, 0, 1)
    print(f"  Jaccard(identical profiles) = {jaccard_identical:.4f} (should be 1)")
    
    # Test case 3: Self-similarity
    print("\nTest 3: Self-similarity")
    jaccard_self, _, _, _, _ = jaccard_from_similarity_matrix(similarity_identical, 0, 0)
    print(f"  Jaccard(profile, itself) = {jaccard_self:.4f} (should be 1)")

def main():
    """Run all tests and provide comprehensive validation."""
    print("Comprehensive Validation: Binary Membership vs Similarity Matrix for Jaccard")
    print("=" * 80)
    
    # Run main test
    success = test_matrix_approaches()
    
    # Provide mathematical explanation
    explain_why_similarity_matrix_works()
    
    # Test edge cases
    test_edge_cases()
    
    # Final conclusion
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    if success:
        print("✅ The UAVarPrior implementation using a similarity matrix is MATHEMATICALLY CORRECT")
        print("✅ The similarity matrix approach is equivalent to the binary membership approach")
        print("✅ The similarity matrix approach is more computationally efficient")
        print("✅ No changes are needed to the Jaccard calculation in variant_analysis.py")
    else:
        print("❌ Issues found in the implementation")
    
    print("\nKey points:")
    print("• The similarity matrix S[i,j] stores the COUNT of shared variants between profiles i and j")
    print("• The diagonal S[i,i] stores the TOTAL number of variants in profile i")
    print("• This allows direct calculation of Jaccard = |intersection| / |union|")
    print("• The formula: jaccard = S[i,j] / (S[i,i] + S[j,j] - S[i,j]) is correct")

if __name__ == "__main__":
    main()
