import pysam
import operator
import numpy as np
from time import time, sleep
import gc
import os

import snvs.constants as c

def has_hard_or_soft_clips_in_the_middle(read):
    cigar = read.cigarstring # e.g. 1S2D3M
    cigar_letters = [] # e.g. ['S', 'D', 'M']
    
    for char in cigar:
        if char.isalpha():
            cigar_letters.append(char)

    for idx, letter in cigar_letters:
        if (letter == 'S' or letter == 'H') and not (idx == 0 or idx == (len(cigar_letters) - 1) ):
            return True

    return False

def fully_hard_or_soft_clipped(read):
    cigar = read.cigarstring # e.g. 1S2D3M
    cigar_letters = [] # e.g. ['S', 'D', 'M']
    
    for char in cigar:
        if char.isalpha():
            cigar_letters.append(char)

    has_non_soft_or_hard_clip_bases = False
    for letter in cigar_letters:
        if letter != 'S' and letter != 'H':
            has_non_soft_or_hard_clip_bases = True

    return not has_non_soft_or_hard_clip_bases

def check_read(read):
    if read.is_unmapped or read.is_duplicate:
        return False

    if read.mapping_quality < c.MIN_READ_MAPPING_QUALITY:
        return False

    # inspired by BadCigarFilter in GATK: https://software.broadinstitute.org/gatk/documentation/tooldocs/current/org_broadinstitute_gatk_engine_filters_BadCigarFilter.php
    # if fully_hard_or_soft_clipped(read):
    #     return False

    # if has_hard_or_soft_clips_in_the_middle(read):
    #     return False

    return True

def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation 
    """
    arr = np.asarray(arr)
    med = np.median(arr)
    return np.median(np.abs(arr - med))

def goldset_pre_filters(sample_name, bamname_n, bamname_t, chrom, start, end, batch_num):
    print(("\n----- Running region: CHROMOSOME %s, %s-%s ------" % (str(chrom), str(start), str(end))))
    
    batch_filename = os.path.join(c.filtering_folder, c.filtering_batches_folder, sample_name, 'chrom%s_batch_%s.csv' % (chrom, str(batch_num)))

    if os.path.exists(batch_filename):
        print(("Skipping %s as it exists" % batch_filename))
        return

    regiontime = time()
    bamfile_n = pysam.AlignmentFile(bamname_n, 'rb')
    bamfile_t = pysam.AlignmentFile(bamname_t, 'rb')

    print("Calculating coverage on normal and tumor...")
    coverage_n = bamfile_n.count_coverage(chrom, start, end + 1)
    coverage_t = bamfile_t.count_coverage(chrom, start, end + 1)

    print("Pre-processing coverage... ")
    # [ (#A, #C, #G, #T), (#A, #C, #G, #T), (#A, #C, #G, #T), ] at each position in normal
    coverage_list_n = [(coverage_n[0][i], coverage_n[1][i], coverage_n[2][i], coverage_n[3][i]) 
        for i in range(len(coverage_n[0]))]
    del coverage_n

    # [ (#A, #C, #G, #T), (#A, #C, #G, #T), (#A, #C, #G, #T), ] at each position in tumor
    coverage_list_t = [(coverage_t[0][i], coverage_t[1][i], coverage_t[2][i], coverage_t[3][i]) 
        for i in range(len(coverage_t[0]))]
    
    del coverage_t

    print("Starting to filter...")
    positions = []

    for pos in range(start, end + 1):
        tumor_reads = bamfile_t.fetch(chrom, pos, pos + 1)
        normal_reads = bamfile_n.fetch(chrom, pos, pos + 1)

        shortest_distances_from_position_to_alignment_end_in_tumor = []
        base_qualities_in_tumor, read_mapping_qualities_in_normal, read_mapping_qualities_in_tumor = [], [], []
        tumor_read_count, reverse_strand_reads, LOW_MAP_QUAL_READS = 0, 0, 0

        for read in normal_reads:
            read_mapping_qualities_in_normal.append(read.mapping_quality)

        for read in tumor_reads:
            if read.is_reverse:
                reverse_strand_reads += 1

            tumor_read_count += 1

            read_mapping_qualities_in_tumor.append(read.mapping_quality)

            if read.mapping_quality < 1:
                LOW_MAP_QUAL_READS += 1

            distance_from_start, distance_from_end = None, None
            
            if read.reference_start:
                distance_from_start = pos - read.reference_start

            if read.reference_end: 
                distance_from_end = (read.reference_end - 1) - pos # reference_end points to one past the last aligned residue. Returns None if not available (read is unmapped or no cigar alignment present

            if distance_from_start and distance_from_end:
                shortest_distances_from_position_to_alignment_end_in_tumor.append(min(distance_from_start, distance_from_end))
            elif distance_from_start:
                shortest_distances_from_position_to_alignment_end_in_tumor.append(distance_from_start)
            elif distance_from_end:
                shortest_distances_from_position_to_alignment_end_in_tumor.append(distance_from_end)
            else:
                continue

            # get base qualities at pos
            for p in read.get_aligned_pairs():
                if p[1] == pos and p[0] is not None: # p[0] implies Deletion
                    base_qualities_in_tumor.append(read.query_qualities[p[0]])

        CANDIDATE_ALLELE_PRESENT = False
        for allele, allele_count in enumerate(coverage_list_t[pos-start]):
            if allele_count >= c.MIN_VARIANT_ALLELE_COUNT and coverage_list_n[pos-start][allele] <= c.MAX_VARIANT_ALLELE_COUNT_IN_CONTROL:
                CANDIDATE_ALLELE_PRESENT = True

        ALLELE__FREQ_FILTER = not CANDIDATE_ALLELE_PRESENT

        STRAND_BIAS_FILTER, LOW_MAP_QUAL_READS_FILTER = False, False
        if tumor_read_count:
            STRAND_BIAS_FILTER = ((float(reverse_strand_reads)/float(tumor_read_count)) < c.MIN_STRAND_BIAS) or ((float(tumor_read_count-reverse_strand_reads)/float(tumor_read_count)) < c.MIN_STRAND_BIAS)
            LOW_MAP_QUAL_READS_FILTER = (float(LOW_MAP_QUAL_READS)/float(tumor_read_count)) > c.MAX_PROPORTION_OF_LOW_MAP_QUAL_READS_AT_VARIANT
 
        MEDIAN_DISTANCE_TO_END_FILTER = np.median(shortest_distances_from_position_to_alignment_end_in_tumor) < c.MIN_DISTANCE_FROM_VARIANT_TO_ALIGNMENT_END_MEDIAN 
        MAD_DISTANCE_TO_END_FILTER = mad(shortest_distances_from_position_to_alignment_end_in_tumor) < c.MIN_DISTANCE_FROM_VARIANT_TO_ALIGNMENT_END_MAD

        MAP_QUAL_DIFF_MEDIAN_FILTER = abs(np.median(read_mapping_qualities_in_tumor) - np.median(read_mapping_qualities_in_normal)) > c.MAX_MAP_QUAL_DIFF_MEDIAN
        VARIANT_MAP_QUAL_MEDIAN_FILTER = np.median(read_mapping_qualities_in_tumor) < c.MIN_VARIANT_MAP_QUAL_MEDIAN
        VARIANT_BASE_QUAL_MEDIAN_FILTER = np.median(base_qualities_in_tumor) < c.MIN_VARIANT_BASE_QUAL_MEDIAN

        if STRAND_BIAS_FILTER or ALLELE__FREQ_FILTER or MEDIAN_DISTANCE_TO_END_FILTER or MAD_DISTANCE_TO_END_FILTER or LOW_MAP_QUAL_READS_FILTER or MAP_QUAL_DIFF_MEDIAN_FILTER or VARIANT_MAP_QUAL_MEDIAN_FILTER or VARIANT_BASE_QUAL_MEDIAN_FILTER:
            continue
        else:
            positions.append(pos)
    
    # create folder for batches for this sample
    if not os.path.exists(os.path.join(c.filtering_folder, c.filtering_batches_folder, sample_name)):
        os.makedirs(os.path.join(c.filtering_folder, c.filtering_batches_folder, sample_name))

    if len(positions):
        # save batch.csv
        with open(batch_filename, 'a') as f:
            for pos in positions:
                f.write('%s\t%s\n' % (chrom, pos))

    print(("Saved batch %s" % batch_filename))

def filter_snvs(candidates_folder, bamname_n, bamname_t, regions, batch_num, output_filename=None):

    # constants
    MIN_BQ  = c.MIN_BASE_QUALITY
    MIN_COV = c.MIN_COVERAGE
    MIN_AF_T = c.MIN_MUTANT_ALLELE_FREQUENCY_IN_TUMOR
    MIN_AR_T = c.MIN_MUTANT_ALLELE_READS_IN_TUMOR
    MAX_AF_N = c.MAX_ALTERNATIVE_ALLELE_FREQUENCY_IN_NORMAL

    output_file = os.path.join(candidates_folder, f"batch_{batch_num}.csv")

    if os.path.exists(output_file):
        print("SNV BATCH COMPLETE:", output_file)
        return

    # t0 = now()
    bamfile_n = pysam.AlignmentFile(bamname_n, "rb")
    bamfile_t = pysam.AlignmentFile(bamname_t, "rb")

    total_positions = 0
    total_alt_candidates = 0
    total_skipped_cov = 0
    total_ties = 0

    with open(output_file, "a") as f:
        for chrom, start, end in regions:
            try:
                coverage_n = bamfile_n.count_coverage(
                    chrom, start, end + 1, quality_threshold=MIN_BQ, read_callback=check_read
                )
            except ValueError:
                alt_chrom = "chrM" if chrom == "MT" else (f"chr{chrom}" if not str(chrom).startswith("chr") else chrom)
                try:
                    coverage_n = bamfile_n.count_coverage(
                        alt_chrom, start, end + 1, quality_threshold=MIN_BQ, read_callback=check_read
                    )
                    chrom = alt_chrom
                except ValueError:
                    print(f"[WARN] Region {chrom}:{start}-{end} missing in normal BAM")
                    continue

            try:
                coverage_t = bamfile_t.count_coverage(
                    chrom, start, end + 1, quality_threshold=MIN_BQ, read_callback=check_read
                )
            except ValueError:
                alt_chrom = "chrM" if chrom == "MT" else (f"chr{chrom}" if not str(chrom).startswith("chr") else chrom)
                try:
                    coverage_t = bamfile_t.count_coverage(
                        alt_chrom, start, end + 1, quality_threshold=MIN_BQ, read_callback=check_read
                    )
                    chrom = alt_chrom
                except ValueError:
                    print(f"[WARN] Region {chrom}:{start}-{end} missing in tumor BAM")
                    continue

            coverage_list_n = np.vstack(coverage_n).astype(np.int32, copy=False)
            coverage_list_t = np.vstack(coverage_t).astype(np.int32, copy=False)
            L = coverage_list_n.shape[1]
            if L == 0:
                continue

            total_positions += L

            coverage_per_pos_n = coverage_list_n.sum(axis=0)
            coverage_per_pos_t = coverage_list_t.sum(axis=0)
            cov_mask = (coverage_per_pos_n >= MIN_COV) & (coverage_per_pos_t >= MIN_COV)
            total_skipped_cov += np.sum(~cov_mask)

            # --- tie-aware alt_mask patch ---
            max_counts = coverage_list_n.max(axis=0)
            ref_mask = coverage_list_n == max_counts   # handle multiallelic ties
            alt_mask = ~ref_mask

            # count tie positions for diagnostics
            ties = (coverage_list_n == max_counts).sum(axis=0) > 1
            total_ties += np.sum(ties)

            # guard against zero coverage
            cov_n_safe = np.where(coverage_per_pos_n == 0, 1, coverage_per_pos_n)
            cov_t_safe = np.where(coverage_per_pos_t == 0, 1, coverage_per_pos_t)

            # use float64 to avoid rounding-out tiny AFs
            alt_freq_t = coverage_list_t.astype(np.float64) / cov_t_safe
            alt_reads_t = coverage_list_t
            alt_freq_n = coverage_list_n.astype(np.float64) / cov_n_safe

            # per-base filters
            tumor_alt_AF_high = alt_freq_t >= MIN_AF_T
            tumor_alt_AR_high = alt_reads_t >= MIN_AR_T
            normal_alt_AF_low = alt_freq_n <= MAX_AF_N

            alt_pass = (
                alt_mask
                & tumor_alt_AF_high
                & tumor_alt_AR_high
                & normal_alt_AF_low
            )
            any_alt_passes = alt_pass.any(axis=0) & cov_mask
            idx = np.nonzero(any_alt_passes)[0]
            total_alt_candidates += idx.size

            if idx.size:
                out = "\n".join(f"{chrom}\t{start + int(i)}" for i in idx) + "\n"
                f.write(out)

        print(
            f"[DEBUG] Processed {total_positions} positions; "
            f"skipped(low cov): {total_skipped_cov}; "
            f"ties in normal: {total_ties}; "
            f"alt candidates written: {total_alt_candidates}"
        )
        print(f"COMPLETED SNV BATCH: {output_file}")


    # for region in regions:
    #     chrom, start, end = region[0], region[1], region[2]

    #     try:
    #         coverage_n = bam_n.count_coverage(chrom, start, end+1, quality_threshold=c.MIN_BASE_QUALITY, read_callback = check_read)
    #     except ValueError:
    #         if chrom == 'MT':
    #             # MT is chrM in hg19
    #             chrom = 'chrM'
    #         else:
    #             chrom = 'chr%s' % chrom

    #         try:
    #             coverage_n = bam_n.count_coverage(chrom, start, end+1, quality_threshold=c.MIN_BASE_QUALITY, read_callback = check_read)
    #         except ValueError:
    #             print("Region does not exist in normal BAM")
    #             return

    #     try:
    #         coverage_t = bam_t.count_coverage(chrom, start, end+1, quality_threshold=c.MIN_BASE_QUALITY, read_callback = check_read)
    #     except ValueError:
    #         if chrom == 'MT':
    #             # MT is chrM in hg19
    #             chrom = 'chrM'
    #         else:
    #             chrom = 'chr%s' % chrom

    #         try:        
    #             coverage_t = bam_t.count_coverage(chrom, start, end+1, quality_threshold=c.MIN_BASE_QUALITY, read_callback = check_read)
    #         except ValueError:
    #             print("Region does not exist in tumor BAM")
    #             return

    #     # [ (#A, #C, #G, #T), (#A, #C, #G, #T), (#A, #C, #G, #T), ] at each position in normal
    #     coverage_list_n = [(coverage_n[0][i], coverage_n[1][i], coverage_n[2][i], coverage_n[3][i]) 
    #         for i in range(len(coverage_n[0]))]
    #     del coverage_n

    #     # [ (#A, #C, #G, #T), (#A, #C, #G, #T), (#A, #C, #G, #T), ] at each position in tumor
    #     coverage_list_t = [(coverage_t[0][i], coverage_t[1][i], coverage_t[2][i], coverage_t[3][i]) 
    #         for i in range(len(coverage_t[0]))]
        
    #     del coverage_t

    #     gc.collect()

    #     map_dict = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
        
    #     for i in range(len(coverage_list_n)):

    #         pos_key = 'chrom%spos%d' % (chrom, start+i)

    #         coverage_of_normal = sum(coverage_list_n[i])
    #         coverage_of_tumor = sum(coverage_list_t[i])

    #         # Skip if either normal or tumor coverage is low
    #         if coverage_of_normal < c.MIN_COVERAGE or coverage_of_tumor < c.MIN_COVERAGE:
    #             continue

    #         # see if a variant allele is found with more than threshold frequency in tumor
    #         # assuming homozygosity
    #         max_frequency_base_in_normal = max(enumerate( coverage_list_n[i] ), key=operator.itemgetter(1))

    #         is_candidate_position = False

    #         """
    #            Filter positions where no alternate allele has freq > c.MIN_MUTANT_ALLELE_FREQUENCY_IN_TUMOR
    #            Filter positions where no alterate allele has more than 2 supporting reads
    #            Filter positions where no alternate allele is found in the normal sample with frequency less than c.MAX_ALTERNATIVE_ALLELE_FREQUENCY_IN_NORMAL    
    #         """

    #         for j in range(len(coverage_list_t[i])):

    #             if j != max_frequency_base_in_normal[0]:
    #                 # checking alternate alleles in tumor

    #                 allele_frequency_high = (coverage_list_t[i][j] / coverage_of_tumor) >= c.MIN_MUTANT_ALLELE_FREQUENCY_IN_TUMOR        
    #                 allele_read_count_high = coverage_list_t[i][j] >= c.MIN_MUTANT_ALLELE_READS_IN_TUMOR
    #                 allele_frequency_low_in_normal = (coverage_list_n[i][j] / coverage_of_normal) <= c.MAX_ALTERNATIVE_ALLELE_FREQUENCY_IN_NORMAL
                
    #                 is_potential_mutation = allele_frequency_high and allele_read_count_high and allele_frequency_low_in_normal

    #                 if is_potential_mutation:
    #                     is_candidate_position = True
    #                     break

    #         if is_candidate_position:
    #             candidates.append((chrom, start + i))

    # create folder for batches for this sample
    #if not os.path.exists(os.path.join(c.filtering_folder, c.filtering_batches_folder, sample_name)):
    #    os.makedirs(os.path.join(c.filtering_folder, c.filtering_batches_folder, sample_name))

    # save batch.csv
    # with open(output_file, 'a') as f:
    #     for pos in candidates:
    #         f.write('%s\t%s\n' % (pos[0], pos[1]))

    # print(('COMPLETED SNV BATCH: ', output_file)) 
