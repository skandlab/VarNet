import os
from os.path import join
from datetime import datetime

# TRAINING DATA

# 43 liver patients
liver_patients_smurf_predictions = '/mnt/projects/krishnak/kiran/SMURF_LIVER_PREDICTIONS'
liver_patients_root_folder = '/mnt/projects/skanderupamj/wgs/data/training/ready.bams/liver/'
liver_patients_normal_bam_file_path = os.path.join(liver_patients_root_folder, '%s-N-ready.bam')
liver_patients_tumor_bam_file_path = os.path.join(liver_patients_root_folder, '%s-T-ready.bam')

# 164 crc patients
crc_patients_smurf_predictions = '/seq/astar/gis/projects/skandera/SMUDL/SMURF_CRC_PREDICTIONS'
crc_patients_smurf_indel_predictions = '/seq/astar/gis/projects/skandera/SMUDL/SMURF_CRC_INDEL_PREDICTIONS'
crc_patients_normal_bam_file_path = '/seq/astar/gis/projects/skandera/training/wgs/colorectal/%s-N-ready.bam'
crc_patients_tumor_bam_file_path = '/seq/astar/gis/projects/skandera/training/wgs/colorectal/%s-T-ready.bam'

# 38 gastric patients
gastric_patients_smurf_predictions = '/seq/astar/gis/projects/skandera/SMUDL/SMURF_GASTRIC_PREDICTIONS/'
gastric_patients_smurf_indel_predictions = '/seq/astar/gis/projects/skandera/SMUDL/SMURF_GASTRIC_INDEL_PREDICTIONS/'
gastric_patients_normal_bam_file_path = '/seq/astar/gis/projects/skandera/training/wgs/gastric/%s-N-ready.bam'
gastric_patients_tumor_bam_file_path = '/seq/astar/gis/projects/skandera/training/wgs/gastric/%s-T-ready.bam'

# 22 lung patients
lung_patients_smurf_predictions = '/mnt/projects/krishnak/kiran/SMURF_LUNG_PREDICTIONS'
lung_bam_files_root_folder = '/home/skanderupamj/projects/wgs/data/luad/'
lung_bam_files_root_folder = '/mnt/projects/skanderupamj/wgs/data/training/ready.bams/lung/'
lung_patients_normal_bam_file_path = os.path.join(lung_bam_files_root_folder, '%s-N-ready.bam')
lung_patients_tumor_bam_file_path = os.path.join(lung_bam_files_root_folder, '%s-T-ready.bam')

# 23 sarcoma patients
sarcoma_patients_smurf_predictions = '/mnt/projects/krishnak/kiran/SMURF_SARCOMA_PREDICTIONS/'
sarcoma_bam_files_root_folder = '/mnt/projects/skanderupamj/wgs/data/training/ready.bams/cfdna/'
sarcoma_patients_normal_bam_file_path = os.path.join(sarcoma_bam_files_root_folder, '%s-N-ready.bam')
sarcoma_patients_tumor_bam_file_path = os.path.join(sarcoma_bam_files_root_folder, '%s-T-ready.bam')

# 6 thyroid patients
thyroid_patients_smurf_predictions = '/mnt/projects/krishnak/kiran/SMURF_THYROID_PREDICTIONS'
thyroid_bam_files_root_folder = '/mnt/projects/skanderupamj/wgs/data/training/ready.bams/thyroid/'
thyroid_patients_normal_bam_file_path = os.path.join(thyroid_bam_files_root_folder, '%s-N-ready.bam')
thyroid_patients_tumor_bam_file_path = os.path.join(thyroid_bam_files_root_folder, '%s-T-ready.bam')

# 60 lymphoma patients
lymphoma_patients_smurf_predictions = '/seq/astar/gis/projects/skandera/SMUDL/SMURF_LYMPHOMA_PREDICTIONS'
lymphoma_patients_smurf_indel_predictions = '/seq/astar/gis/projects/skandera/SMUDL/SMURF_LYMPHOMA_INDEL_PREDICTIONS'
lymphoma_patients_normal_bam_file_path = '/seq/astar/gis/projects/skandera/training/wgs/lymphoma/%s-N-ready.bam'
lymphoma_patients_tumor_bam_file_path = '/seq/astar/gis/projects/skandera/training/wgs/lymphoma/%s-T-ready.bam'

# FFPE samples
ffpe_vcfs_folder = '/home/users/astar/gis/krishnak/scratch_mnt/ffpe_parse_vcfs' 
# FFPE_samples = ['A150', 'A575', 'A592', 'A623', 'A624', 'A649', 'A710', 'A883']
#FFPE_samples = ['A001','A014','A112','A123','A169','A435','A512','A570','A599','A611','A772']

# WES lung internal
FFPE_SAMPLES = ['A001','A014','A112','A150','A435','A512','A570','A599','A611'] # note: remaining ffpe samples were not taken at the same time as frozen
ffpe_bams_folder = '/scratch/users/astar/gis/krishnak/internal_FFPE_WES'

mutect2_calls_on_ffpe = [ ('%s-FFPE-mutect2-pass-calls' % x, os.path.join('/home/project/13002420/varnet/', '%s-FFPE-candidates-mutect2-pass-calls' % x, 'candidates', 'snvs', 'Positions.csv'), os.path.join(ffpe_vcfs_folder, x, '%s-FFPE.varnet.frzn.common.with.smurf_snvs.txt' % x), os.path.join(ffpe_bams_folder, '%s-FFPE-EH-N-ready.bam' % x), os.path.join(ffpe_bams_folder, '%s-FFPE-EH-T-ready.bam' % x)) for x in FFPE_SAMPLES ]

strelka2_calls_on_ffpe = [ ('%s-FFPE-strelka2-pass-calls' % x, os.path.join('/home/project/13002420/varnet/', '%s-FFPE-candidates-strelka2-pass-calls' % x, 'candidates', 'snvs', 'Positions.csv'), os.path.join(ffpe_vcfs_folder, x, '%s-FFPE.varnet.frzn.common.with.smurf_snvs.txt' % x), os.path.join(ffpe_bams_folder, '%s-FFPE-EH-N-ready.bam' % x), os.path.join(ffpe_bams_folder, '%s-FFPE-EH-T-ready.bam' % x)) for x in FFPE_SAMPLES ]

# WGS GDC lung sample list format [ [sample_name, normal_bam, tumor_bam], ... ]
#FFPE_SAMPLES = [['Normal-TCGA-44-2656-10A__Tumor-TCGA-44-2656', '/home/users/astar/gis/krishnak/scratch_mnt/GDC_bams/c3373dbc-2d44-47ab-8ce7-811ec68a13da/f9538cdc-2449-4260-af50-e41dc1ecab6a_wgs_gdc_realn.bam', '/home/users/astar/gis/krishnak/scratch_mnt/GDC_bams/1027f76d-b2a9-4f0a-b173-d52d81eef832/a8088515-6622-42ba-93b0-84229c023c6f_wgs_gdc_realn.bam'], ['Normal-TCGA-44-2666-10A__Tumor-TCGA-44-2666', '/home/users/astar/gis/krishnak/scratch_mnt/GDC_bams/ed07323f-b2bc-4ce4-bb90-baaad2f2a5bf/e8291a63-47be-48a8-8c42-d41f3e4dd6a4_wgs_gdc_realn.bam', '/home/users/astar/gis/krishnak/scratch_mnt/GDC_bams/1c797505-5fd7-4af2-ba44-2241b8249daa/e93df4f6-b66e-42a2-b20d-c60b6e550b9d_wgs_gdc_realn.bam'], ['Normal-TCGA-44-2668-10A__Tumor-TCGA-44-2668', '/home/users/astar/gis/krishnak/scratch_mnt/GDC_bams/cec0acc0-c78f-4dfe-b792-dc62ecc0549a/8c9b8c19-d9c0-4b77-96b5-a2edb8a0d95a_wgs_gdc_realn.bam', '/home/users/astar/gis/krishnak/scratch_mnt/GDC_bams/25857522-1d0e-4e6e-b5aa-8d3b9780c726/229b83ac-c90d-4807-8b2d-7e641ad0d967_wgs_gdc_realn.bam'], ['Normal-TCGA-44-3917-10A__Tumor-TCGA-44-3917', '/home/users/astar/gis/krishnak/scratch_mnt/GDC_bams/0665ef25-3043-45c1-a6d6-0aca220916a5/71e3a02d-4557-4ff7-b759-56f7fb43dcb3_wgs_gdc_realn.bam', '/home/users/astar/gis/krishnak/scratch_mnt/GDC_bams/0d35804c-d771-40e4-9ca1-830bf1e07255/5adf6723-471c-44ae-a7ae-004c019d3664_wgs_gdc_realn.bam'], ['Normal-TCGA-44-3918-11A__Tumor-TCGA-44-3918', '/home/users/astar/gis/krishnak/scratch_mnt/GDC_bams/9ce3acf6-56c9-4014-92cb-4edfb722032e/46149026-2fad-4fcd-a4e5-b02224fed763_wgs_gdc_realn.bam', '/home/users/astar/gis/krishnak/scratch_mnt/GDC_bams/d0954874-c1ad-4c67-9009-d8fb505ac7f5/ec643748-14f8-4274-879f-c0813193a45c_wgs_gdc_realn.bam'], ['Normal-TCGA-44-3918-10A__Tumor-TCGA-44-3918', '/home/users/astar/gis/krishnak/scratch_mnt/GDC_bams/df42c9e6-0d6a-4c10-a6af-91fd074f163b/55f81e04-3dee-4aa0-93a7-7789d1a1d7ef_wgs_gdc_realn.bam', '/home/users/astar/gis/krishnak/scratch_mnt/GDC_bams/d0954874-c1ad-4c67-9009-d8fb505ac7f5/ec643748-14f8-4274-879f-c0813193a45c_wgs_gdc_realn.bam'], ['Normal-TCGA-44-4112-11A__Tumor-TCGA-44-4112', '/home/users/astar/gis/krishnak/scratch_mnt/GDC_bams/c6243ff0-bf3f-42e6-af0b-df96c621aa36/161ab88f-51df-41df-b38c-c1dc9764dc82_wgs_gdc_realn.bam', '/home/users/astar/gis/krishnak/scratch_mnt/GDC_bams/c06b3853-6c86-41f1-81ea-123ad6b03534/55c0e17b-a9a5-46a5-a4ac-d26cbd83e7b1_wgs_gdc_realn.bam'], ['Normal-TCGA-44-4112-10A__Tumor-TCGA-44-4112', '/home/users/astar/gis/krishnak/scratch_mnt/GDC_bams/80fd0f6a-095c-4b80-9f91-e8334cbe32da/d59902b3-f49d-4561-9a7c-6b2545588073_wgs_gdc_realn.bam', '/home/users/astar/gis/krishnak/scratch_mnt/GDC_bams/c06b3853-6c86-41f1-81ea-123ad6b03534/55c0e17b-a9a5-46a5-a4ac-d26cbd83e7b1_wgs_gdc_realn.bam'], ['Normal-TCGA-44-5645-10A__Tumor-TCGA-44-5645', '/home/users/astar/gis/krishnak/scratch_mnt/GDC_bams/0ed82e37-2bc0-4c86-8ce4-4db20b735e82/78734754-0680-489a-886b-2f5fce472560_wgs_gdc_realn.bam', '/home/users/astar/gis/krishnak/scratch_mnt/GDC_bams/5b53713a-3d04-4629-852d-00aeeadd9362/9621f560-ac27-4563-9798-f982ace7944b_wgs_gdc_realn.bam'], ['Normal-TCGA-44-6146-11A__Tumor-TCGA-44-6146', '/home/users/astar/gis/krishnak/scratch_mnt/GDC_bams/475dc9c5-3640-4af8-8f46-9589aad665c6/03a2138c-b699-47d1-a5eb-161399020946_wgs_gdc_realn.bam', '/home/users/astar/gis/krishnak/scratch_mnt/GDC_bams/0e59a472-9eb3-4100-9c01-02389a923656/6051356a-2f32-44f7-a36e-13ad6eef6f00_wgs_gdc_realn.bam'], ['Normal-TCGA-44-6146-10A__Tumor-TCGA-44-6146', '/home/users/astar/gis/krishnak/scratch_mnt/GDC_bams/9ff09396-532f-4541-990f-f0f82b0c993d/8d8c3fbf-204b-4a7a-942a-6a6ce2e0470c_wgs_gdc_realn.bam', '/home/users/astar/gis/krishnak/scratch_mnt/GDC_bams/0e59a472-9eb3-4100-9c01-02389a923656/6051356a-2f32-44f7-a36e-13ad6eef6f00_wgs_gdc_realn.bam'], ['Normal-TCGA-44-6147-11A__Tumor-TCGA-44-6147', '/home/users/astar/gis/krishnak/scratch_mnt/GDC_bams/13ff444a-c692-47b1-9a50-8383bbd278fe/503a3086-3453-4aae-987b-019fbcb13272_wgs_gdc_realn.bam', '/home/users/astar/gis/krishnak/scratch_mnt/GDC_bams/bdf22e66-fa1e-4e99-aed2-4e53189b3c59/6dabceb4-789d-49c2-8040-8900ccdbf700_wgs_gdc_realn.bam'], ['Normal-TCGA-44-6147-10A__Tumor-TCGA-44-6147', '/home/users/astar/gis/krishnak/scratch_mnt/GDC_bams/276b4b44-784d-4a34-8c6b-e8fd3b15b232/ca27bdf5-676b-42c3-8e49-ebd4f86c1e2e_wgs_gdc_realn.bam', '/home/users/astar/gis/krishnak/scratch_mnt/GDC_bams/bdf22e66-fa1e-4e99-aed2-4e53189b3c59/6dabceb4-789d-49c2-8040-8900ccdbf700_wgs_gdc_realn.bam'], ['Normal-TCGA-44-6775-11A__Tumor-TCGA-44-6775', '/home/users/astar/gis/krishnak/scratch_mnt/GDC_bams/9f5f398c-3e05-4b34-a210-682846ad274f/13d5fbfa-d5a3-4195-9756-91151ecaa5cf_wgs_gdc_realn.bam', '/home/users/astar/gis/krishnak/scratch_mnt/GDC_bams/4867f501-6180-41ea-a332-401021e065ca/e23f7ef6-cf91-4812-afda-02d30b14af4f_wgs_gdc_realn.bam'], ['Normal-TCGA-44-6775-10A__Tumor-TCGA-44-6775', '/home/users/astar/gis/krishnak/scratch_mnt/GDC_bams/bd85d3d3-e06b-4e19-9a11-e7a0bebfea07/b76d5b8b-81e7-4f5e-9ceb-29f3f47db3b0_wgs_gdc_realn.bam', '/home/users/astar/gis/krishnak/scratch_mnt/GDC_bams/4867f501-6180-41ea-a332-401021e065ca/e23f7ef6-cf91-4812-afda-02d30b14af4f_wgs_gdc_realn.bam'], ['Normal-TCGA-BL-A0C8-10A__Tumor-TCGA-BL-A0C8', '/home/users/astar/gis/krishnak/scratch_mnt/GDC_bams/b234d209-ed08-41f9-80ff-081ffaab39d1/49aa01dd-8cdc-47f6-9d09-87d4862fa071_wgs_gdc_realn.bam', '/home/users/astar/gis/krishnak/scratch_mnt/GDC_bams/594954b9-73b4-4d21-983a-a85f2921a3f1/b681fd85-05fc-4667-bed2-6b7b35fd4860_wgs_gdc_realn.bam'], ['Normal-TCGA-BL-A13J-11A__Tumor-TCGA-BL-A13J', '/home/users/astar/gis/krishnak/scratch_mnt/GDC_bams/f20dfc6a-da2d-453f-a91e-52cf55122271/9b4ffd23-8b53-4dfc-a98e-e4c70f7680f0_wgs_gdc_realn.bam', '/home/users/astar/gis/krishnak/scratch_mnt/GDC_bams/3586906b-0530-46c2-9771-d043c2103ddb/a9a3c98d-6e7e-4a2c-afb1-cf9b220d78bc_wgs_gdc_realn.bam'], ['Normal-TCGA-BL-A13J-10A__Tumor-TCGA-BL-A13J', '/home/users/astar/gis/krishnak/scratch_mnt/GDC_bams/439dfda5-769d-4077-b531-aae7c6bb2fe1/e34479b8-834f-4747-984b-e77272f1379d_wgs_gdc_realn.bam', '/home/users/astar/gis/krishnak/scratch_mnt/GDC_bams/3586906b-0530-46c2-9771-d043c2103ddb/a9a3c98d-6e7e-4a2c-afb1-cf9b220d78bc_wgs_gdc_realn.bam']]
#ffpe_bams_folder = '/home/users/astar/gis/krishnak/scratch_mnt/ffpe_parse_vcfs'

# FFPE TRAINING: SEQC2 + TCGA 
ffpe_wgs_wes_training_tcga_samples = ['Normal-TCGA-44-2656-10A__Tumor-TCGA-44-2656-01B-WH', 'Normal-TCGA-44-2666-10A__Tumor-TCGA-44-2666-01B-WH', 'Normal-TCGA-44-2668-10A__Tumor-TCGA-44-2668-01B-WH', 'Normal-TCGA-44-3917-10A__Tumor-TCGA-44-3917-01B-WH', 'Normal-TCGA-44-3918-10A__Tumor-TCGA-44-3918-01B-WH', 'Normal-TCGA-44-3918-11A__Tumor-TCGA-44-3918-01B-WH', 'Normal-TCGA-44-4112-10A__Tumor-TCGA-44-4112-01B-WH', 'Normal-TCGA-44-4112-11A__Tumor-TCGA-44-4112-01B-WH', 'Normal-TCGA-44-6146-10A__Tumor-TCGA-44-6146-01B-WH', 'Normal-TCGA-44-6146-11A__Tumor-TCGA-44-6146-01B-WH', 'Normal-TCGA-44-6147-10A__Tumor-TCGA-44-6147-01B-WH', 'Normal-TCGA-44-6147-11A__Tumor-TCGA-44-6147-01B-WH', 'Normal-TCGA-44-6775-10A__Tumor-TCGA-44-6775-01C-WH', 'Normal-TCGA-44-6775-11A__Tumor-TCGA-44-6775-01C-WH', 'Normal-TCGA-BL-A0C8-10A__Tumor-TCGA-BL-A0C8-01B-WH', 'Normal-TCGA-BL-A13J-10A__Tumor-TCGA-BL-A13J-01B-WH', 'Normal-TCGA-BL-A13J-11A__Tumor-TCGA-BL-A13J-01B-WH']
# for TCGA, use BAMs from the bcbio runs
ffpe_wgs_wes_training  = [[_, join('/home/users/astar/gis/krishnak/scratch_mnt/ffpe_parse_vcfs/training_positions/',_+'.csv'), '/home/users/astar/gis/simngl/scratch/F4PE-WGS-bcbio/thirty-six/completed/%s/bcbio_final/%s-N/%s-N-ready.bam'%(_,_,_),  '/home/users/astar/gis/simngl/scratch/F4PE-WGS-bcbio/thirty-six/completed/%s/bcbio_final/%s-T/%s-T-ready.bam'%(_,_,_)] for _ in ffpe_wgs_wes_training_tcga_samples] 

# use frozen normal for SEQC2
ffpe_wgs_training_seqc2_samples = [['seqc2_frzn_norealign_N_ffpe_T_1h','FFG_IL_T_1h'], ['seqc2_frzn_norealign_N_ffpe_T_2h','FFG_IL_T_2h'], ['seqc2_frzn_norealign_N_ffpe_T_6h','FFG_IL_T_6h'], ['seqc2_frzn_norealign_N_ffpe_T_24h','FFG_IL_T_24h']]
ffpe_wgs_wes_training = [[_[0], join('/home/users/astar/gis/krishnak/scratch_mnt/ffpe_parse_vcfs/training_positions/',_[0]+'.csv'), '/home/users/astar/gis/krishnak/scratch_mnt/seqc2/without_indel_realignment/WGS_IL_1-N-ready.bam', '/home/users/astar/gis/krishnak/scratch_mnt/SEQC2_FFPE_WGS/%s.bwa.dedup.bam' % _[1]] for _ in ffpe_wgs_training_seqc2_samples]

# SEQC2 FFPE WGS (ground truth v2)
ffpe_wgs_training_seqc2_samples = [['seqc2_frzn_norealign_N_ffpe_T_1h_v2','FFG_IL_T_1h'], ['seqc2_frzn_norealign_N_ffpe_T_2h_v2','FFG_IL_T_2h'], ['seqc2_frzn_norealign_N_ffpe_T_6h_v2','FFG_IL_T_6h'], ['seqc2_frzn_norealign_N_ffpe_T_24h_v2','FFG_IL_T_24h']]
ffpe_wgs_wes_training = [[_[0], join('/home/users/astar/gis/krishnak/scratch_mnt/ffpe_parse_vcfs/training_positions/',_[0]+'.csv'), '/home/users/astar/gis/krishnak/scratch_mnt/seqc2/without_indel_realignment/WGS_IL_1-N-ready.bam', '/home/users/astar/gis/krishnak/scratch_mnt/SEQC2_FFPE_WGS/%s.bwa.dedup.bam' % _[1]] for _ in ffpe_wgs_training_seqc2_samples]

# SEQC2 FFPE WGS (ground truth v2 + read orientation)
ffpe_wgs_training_seqc2_samples = [['seqc2_frzn_norealign_N_ffpe_T_1h_v2','FFG_IL_T_1h'], ['seqc2_frzn_norealign_N_ffpe_T_2h_v2','FFG_IL_T_2h'], ['seqc2_frzn_norealign_N_ffpe_T_6h_v2','FFG_IL_T_6h'], ['seqc2_frzn_norealign_N_ffpe_T_24h_v2','FFG_IL_T_24h']]
ffpe_wgs_wes_training = [[_[0] + '_RO', join('/home/users/astar/gis/krishnak/scratch_mnt/ffpe_parse_vcfs/training_positions/',_[0]+'.csv'), '/home/users/astar/gis/krishnak/scratch_mnt/seqc2/without_indel_realignment/WGS_IL_1-N-ready.bam', '/home/users/astar/gis/krishnak/scratch_mnt/SEQC2_FFPE_WGS/%s.bwa.dedup.bam' % _[1]] for _ in ffpe_wgs_training_seqc2_samples]

# SEQC2 FFPE WGS (ground truth v3)
#ffpe_wgs_training_seqc2_samples = [['seqc2_frzn_norealign_N_ffpe_T_1h_v3','FFG_IL_T_1h'], ['seqc2_frzn_norealign_N_ffpe_T_2h_v3','FFG_IL_T_2h'], ['seqc2_frzn_norealign_N_ffpe_T_6h_v3','FFG_IL_T_6h'], ['seqc2_frzn_norealign_N_ffpe_T_24h_v3','FFG_IL_T_24h']]
#ffpe_wgs_wes_training = [[_[0], join('/home/users/astar/gis/krishnak/scratch_mnt/ffpe_parse_vcfs/training_positions/',_[0]+'.csv'), '/home/users/astar/gis/krishnak/scratch_mnt/seqc2/without_indel_realignment/WGS_IL_1-N-ready.bam', '/home/users/astar/gis/krishnak/scratch_mnt/SEQC2_FFPE_WGS/%s.bwa.dedup.bam' % _[1]] for _ in ffpe_wgs_training_seqc2_samples]

# SEQC2 FFPE WES
#ffpe_wes_training_seqc2_samples = [['seqc2_wes_frzn_N_ffpe_1h_T','SEQC2_FFX_IL_T_1h_1_FFPE-EH'], ['seqc2_wes_frzn_N_ffpe_2h_T','SEQC2_FFX_IL_T_2h_1_FFPE-EH'], ['seqc2_wes_frzn_N_ffpe_6h_T','SEQC2_FFX_IL_T_6h_1_FFPE-EH'], ['seqc2_wes_frzn_N_ffpe_24h_T','SEQC2_FFX_IL_T_24h_1_FFPE-EH']]
#ffpe_wgs_wes_training = [[_[0], join('/home/users/astar/gis/krishnak/scratch_mnt/ffpe_parse_vcfs/training_positions/',_[0]+'.csv'), '/scratch/users/astar/gis/simngl/SEQC2-Kiran/results/SEQC2-bams/%s-N-ready.bam' % _[1], '/scratch/users/astar/gis/simngl/SEQC2-Kiran/results/SEQC2-bams/%s-T-ready.bam' % _[1]] for _ in ffpe_wes_training_seqc2_samples]

# SEQC2 FFPE WGS (varnet_negatives)
#ffpe_wgs_training_seqc2_samples = [['seqc2_frzn_norealign_N_ffpe_T_1h.varnet_negatives','FFG_IL_T_1h'], ['seqc2_frzn_norealign_N_ffpe_T_2h.varnet_negatives','FFG_IL_T_2h'], ['seqc2_frzn_norealign_N_ffpe_T_6h.varnet_negatives','FFG_IL_T_6h'], ['seqc2_frzn_norealign_N_ffpe_T_24h.varnet_negatives','FFG_IL_T_24h']]
#ffpe_wgs_wes_training = [[_[0], join('/home/users/astar/gis/krishnak/scratch_mnt/ffpe_parse_vcfs/training_positions/',_[0]+'.csv'), '/home/users/astar/gis/krishnak/scratch_mnt/seqc2/without_indel_realignment/WGS_IL_1-N-ready.bam', '/home/users/astar/gis/krishnak/scratch_mnt/SEQC2_FFPE_WGS/%s.bwa.dedup.bam' % _[1]] for _ in ffpe_wgs_training_seqc2_samples]

# SEQC2 FFPE WGS (mutect2_negatives)
#ffpe_wgs_training_seqc2_samples = [['seqc2_frzn_norealign_N_ffpe_T_1h.mutect2_negatives','FFG_IL_T_1h'], ['seqc2_frzn_norealign_N_ffpe_T_2h.mutect2_negatives','FFG_IL_T_2h'], ['seqc2_frzn_norealign_N_ffpe_T_6h.mutect2_negatives','FFG_IL_T_6h'], ['seqc2_frzn_norealign_N_ffpe_T_24h.mutect2_negatives','FFG_IL_T_24h']]
#ffpe_wgs_wes_training += [[_[0], join('/home/users/astar/gis/krishnak/scratch_mnt/ffpe_parse_vcfs/training_positions/',_[0]+'.csv'), '/home/users/astar/gis/krishnak/scratch_mnt/seqc2/without_indel_realignment/WGS_IL_1-N-ready.bam', '/home/users/astar/gis/krishnak/scratch_mnt/SEQC2_FFPE_WGS/%s.bwa.dedup.bam' % _[1]] for _ in ffpe_wgs_training_seqc2_samples]

## <start> SEQC2 FFPE WES + WGS v4 (tumor-only, SNV and INDEL combined)
ffpe_wgs_training_seqc2_samples = [['seqc2_frzn_norealign_N_ffpe_T_1h_v4','FFG_IL_T_1h'], ['seqc2_frzn_norealign_N_ffpe_T_2h_v4','FFG_IL_T_2h'], ['seqc2_frzn_norealign_N_ffpe_T_6h_v4','FFG_IL_T_6h'], ['seqc2_frzn_norealign_N_ffpe_T_24h_v4','FFG_IL_T_24h']]
ffpe_wgs_training_seqc2_samples = [['seqc2_frzn_norealign_N_ffpe_T_24h_v4','FFG_IL_T_24h']]
ffpe_wgs_wes_training = [[_[0], join('/scratch/users/astar/gis/krishnak/ffpe_parse_vcfs/training_positions/',_[0]+'.csv'), '/scratch/users/astar/gis/krishnak/seqc2/without_indel_realignment/WGS_IL_1-N-ready.bam', '/scratch/users/astar/gis/krishnak/SEQC2_FFPE_WGS/%s.bwa.dedup.bam' % _[1], '/scratch/users/astar/gis/krishnak/project/GRCh38.d1.vd1.fa'] for _ in ffpe_wgs_training_seqc2_samples]

# SEQC2 FFPE WES (SNV + INDEL) v4
ffpe_wes_training_seqc2_samples = [['seqc2_wes_frzn_N_ffpe_1h_T_v4','SEQC2_FFX_IL_T_1h_1_FFPE-EH'], ['seqc2_wes_frzn_N_ffpe_2h_T_v4','SEQC2_FFX_IL_T_2h_1_FFPE-EH'], ['seqc2_wes_frzn_N_ffpe_6h_T_v4','SEQC2_FFX_IL_T_6h_1_FFPE-EH'], ['seqc2_wes_frzn_N_ffpe_24h_T_v4','SEQC2_FFX_IL_T_24h_1_FFPE-EH']]
ffpe_wgs_wes_training += [[_[0], join('/scratch/users/astar/gis/krishnak/ffpe_parse_vcfs/training_positions/',_[0]+'.csv'), '/scratch/users/astar/gis/krishnak/SEQC2_FFPE_WXS/bams/%s-N-ready.bam' % _[1], '/scratch/users/astar/gis/krishnak/SEQC2_FFPE_WXS/bams/%s-T-ready.bam' % _[1], '/scratch/users/astar/gis/krishnak/hg38.fa'] for _ in ffpe_wes_training_seqc2_samples]

# tumor dilution of FFPE 24H WGS and WES SEQC2 samples (only using the positive mutation sites for diluted samples as the FFPE artifacts should not be different for a low purity sample. The FFPE damage occurs uniformly across tumor and normal cells)
# the mixing has been done with the matched FFPE normal sample. see ~/scratch/dilute_samples
# 50% purity
# WGS
ffpe_wgs_wes_training += [['seqc2_ffpe_T_24h_subsampled_0.5_plus_ffpe_N_24h_subsampled_0.5_v4', '/scratch/users/astar/gis/krishnak/ffpe_parse_vcfs/training_positions/seqc2_ffpe_T_24h_subsampled_0.5_plus_ffpe_N_24h_subsampled_0.5_v4.csv', '/scratch/users/astar/gis/krishnak/SEQC2_FFPE_WGS/FFG_IL_N_24h.bwa.dedup.bam', '/scratch/users/astar/gis/krishnak/SEQC2_FFPE_WGS/FFG_IL_T_24h.bwa.dedup.subsampled_0.5.plus.FFG_IL_N_24h.bwa.dedup.subsampled_0.5.bam', '/scratch/users/astar/gis/krishnak/hg38.fa']]
# WXS
ffpe_wgs_wes_training += [['seqc2_wes_ffpe_T_24h_subsampled_0.5_plus_ffpe_24h_N_subsampled_0.5_v4', '/scratch/users/astar/gis/krishnak/ffpe_parse_vcfs/training_positions/seqc2_wes_ffpe_T_24h_subsampled_0.5_plus_ffpe_24h_N_subsampled_0.5_v4.csv', '/scratch/users/astar/gis/krishnak/SEQC2_FFPE_WXS/bams/SEQC2_FFX_IL_T_24h_1_FFPE-EH-N-ready.bam', '/scratch/users/astar/gis/krishnak/SEQC2_FFPE_WXS/bams/SEQC2_FFX_IL_T_24h_1_FFPE-EH-T-ready.subsampled_0.5.plus.SEQC2_FFX_IL_T_24h_1_FFPE-EH-N-ready.subsampled_0.5.bam', '/scratch/users/astar/gis/krishnak/hg38.fa']]

# 40% purity
# WGS
ffpe_wgs_wes_training += [['seqc2_ffpe_T_24h_subsampled_0.4_plus_ffpe_N_24h_subsampled_0.6_v4', '/scratch/users/astar/gis/krishnak/ffpe_parse_vcfs/training_positions/seqc2_ffpe_T_24h_subsampled_0.4_plus_ffpe_N_24h_subsampled_0.6_v4.csv', '/scratch/users/astar/gis/krishnak/SEQC2_FFPE_WGS/FFG_IL_N_24h.bwa.dedup.bam', '/scratch/users/astar/gis/krishnak/SEQC2_FFPE_WGS/FFG_IL_T_24h.bwa.dedup.subsampled_0.4.plus.FFG_IL_N_24h.bwa.dedup.subsampled_0.6.bam', '/scratch/users/astar/gis/krishnak/hg38.fa']]
# WXS
ffpe_wgs_wes_training += [['seqc2_wes_ffpe_T_24h_subsampled_0.4_plus_ffpe_24h_N_subsampled_0.6_v4', '/scratch/users/astar/gis/krishnak/ffpe_parse_vcfs/training_positions/seqc2_wes_ffpe_T_24h_subsampled_0.4_plus_ffpe_24h_N_subsampled_0.6_v4.csv', '/scratch/users/astar/gis/krishnak/SEQC2_FFPE_WXS/bams/SEQC2_FFX_IL_T_24h_1_FFPE-EH-N-ready.bam', '/scratch/users/astar/gis/krishnak/SEQC2_FFPE_WXS/bams/SEQC2_FFX_IL_T_24h_1_FFPE-EH-T-ready.subsampled_0.4.plus.SEQC2_FFX_IL_T_24h_1_FFPE-EH-N-ready.subsampled_0.6.bam', '/scratch/users/astar/gis/krishnak/hg38.fa']]

# 30% purity
# WGS
ffpe_wgs_wes_training += [['seqc2_ffpe_T_24h_subsampled_0.3_plus_ffpe_N_24h_subsampled_0.7_v4', '/scratch/users/astar/gis/krishnak/ffpe_parse_vcfs/training_positions/seqc2_ffpe_T_24h_subsampled_0.3_plus_ffpe_N_24h_subsampled_0.7_v4.csv', '/scratch/users/astar/gis/krishnak/SEQC2_FFPE_WGS/FFG_IL_N_24h.bwa.dedup.bam', '/scratch/users/astar/gis/krishnak/SEQC2_FFPE_WGS/FFG_IL_T_24h.bwa.dedup.subsampled_0.3.plus.FFG_IL_N_24h.bwa.dedup.subsampled_0.7.bam', '/scratch/users/astar/gis/krishnak/hg38.fa']]
# WXS
ffpe_wgs_wes_training += [['seqc2_wes_ffpe_T_24h_subsampled_0.3_plus_ffpe_24h_N_subsampled_0.7_v4', '/scratch/users/astar/gis/krishnak/ffpe_parse_vcfs/training_positions/seqc2_wes_ffpe_T_24h_subsampled_0.3_plus_ffpe_24h_N_subsampled_0.7_v4.csv', '/scratch/users/astar/gis/krishnak/SEQC2_FFPE_WXS/bams/SEQC2_FFX_IL_T_24h_1_FFPE-EH-N-ready.bam', '/scratch/users/astar/gis/krishnak/SEQC2_FFPE_WXS/bams/SEQC2_FFX_IL_T_24h_1_FFPE-EH-T-ready.subsampled_0.3.plus.SEQC2_FFX_IL_T_24h_1_FFPE-EH-N-ready.subsampled_0.7.bam', '/scratch/users/astar/gis/krishnak/hg38.fa']]

## </start> SEQC2 FFPE WES + WGS v4 (tumor-only, SNV and INDEL combined)

# <start> FFPE bcbio-1pct calls made by varnet, mutect2 and strelka2 on FFPE samples
ffpe_lung_cohort_calls = []

# 0-indexed snv and indel combined. created using ~/scratch/ffpe_parse_vcfs/create_ffpe_ground_truth.py
for _ in FFPE_SAMPLES:
	ffpe_lung_cohort_calls.append([_, join('/scratch/users/astar/gis/krishnak/ffpe_parse_vcfs/bcbio-1pct-ground-truths/%s_FFPE_varnet_0_indexed_snvs_indels.csv' % _), join(ffpe_bams_folder, '%s-FFPE-EH-N-ready.bam' % _), join(ffpe_bams_folder, '%s-FFPE-EH-T-ready.bam' % _), '/scratch/users/astar/gis/krishnak/hg38.fa'])
	ffpe_lung_cohort_calls.append([_, join('/scratch/users/astar/gis/krishnak/ffpe_parse_vcfs/bcbio-1pct-ground-truths/%s_FFPE_mutect2_0_indexed_snvs_indels.csv' % _), join(ffpe_bams_folder, '%s-FFPE-EH-N-ready.bam' % _), join(ffpe_bams_folder, '%s-FFPE-EH-T-ready.bam' % _), '/scratch/users/astar/gis/krishnak/hg38.fa'])
	ffpe_lung_cohort_calls.append([_, join('/scratch/users/astar/gis/krishnak/ffpe_parse_vcfs/bcbio-1pct-ground-truths/%s_FFPE_strelka2_0_indexed_snvs_indels.csv' % _), join(ffpe_bams_folder, '%s-FFPE-EH-N-ready.bam' % _), join(ffpe_bams_folder, '%s-FFPE-EH-T-ready.bam' % _), '/scratch/users/astar/gis/krishnak/hg38.fa'])
# </start> FFPE bcbio-1pct calls made by varnet, mutect2 and strelka2 on FFPE samples

BALANCE_CANCER_TYPES_IN_TRAINING = False

def get_crc_patient_bam_file_paths(sample_name):
	with open(crc_patients_bam_info_file) as f:
		for line in f:
			if sample_name in line:
				split_line = line.split('\t')
				normal_bam_file, tumor_bam_file = split_line[4], split_line[2]
				return (os.path.join(crc_patients_bam_files_root_path, normal_bam_file), os.path.join(crc_patients_bam_files_root_path, tumor_bam_file))

def get_gastric_patient_bam_file_paths(sample_name):
	gastric_bams = [ x for x in os.listdir(gastric_bam_files_root_folder) if x.endswith('.bam') ]
	normal_bam_file, tumor_bam_file = None, None

	for bam in gastric_bams:
		if 'N' + sample_name in bam:
			normal_bam_file = os.path.join(gastric_bam_files_root_folder, bam)
		if 'T' + sample_name in bam:
			tumor_bam_file = os.path.join(gastric_bam_files_root_folder, bam)

	assert normal_bam_file != None and tumor_bam_file != None

	print(("Sample name %s" % sample_name))
	print(("Normal BAM %s" % normal_bam_file))
	print(("Tumor BAM %s" % tumor_bam_file))

	return (normal_bam_file, tumor_bam_file)

def check_if_crc_patient_has_indexed_bam(sample_name):
	normal_bam, tumor_bam = get_crc_patient_bam_file_paths(sample_name)
	return os.path.exists(normal_bam + '.bai') and os.path.exists(tumor_bam + '.bai')

goldset_files = [

('icgc_cll', '/home/krishnak/smudl/goldset_files/truth1_positions.txt', '/mnt/projects/huangwt/wgs/Results-SMuRF/Real-bcbio103-samples/2015-07-31_icgc_cll/snv-allpredictions.txt', '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_cll-N.bam', '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_cll-T.bam'),

('icgc_mbl', '/home/krishnak/smudl/goldset_files/truth2_positions.txt', '/mnt/projects/huangwt/wgs/Results-SMuRF/Real-bcbio103-samples/2015-07-31_icgc_mbl/snv-allpredictions.txt', '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_mbl-N.bam', '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_mbl-T.bam'),

('icgc_cll-T40', '/home/krishnak/smudl/goldset_files/truth1_positions.txt', '/mnt/projects/huangwt/wgs/Results-SMuRF/Real-bcbio103-samples/2015-07-31_icgc_cll_T40/snv-allpredictions.txt', '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_cll-N.bam', '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_cll-T40.bam'),

('icgc_mbl-T40', '/home/krishnak/smudl/goldset_files/truth2_positions.txt', '/mnt/projects/huangwt/wgs/Results-SMuRF/Real-bcbio103-samples/2015-07-31_icgc_mbl_T40/snv-allpredictions.txt', '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_mbl-N.bam', '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_mbl-T40.bam'),

('icgc_cll-T20', '/home/krishnak/smudl/goldset_files/truth1_positions.txt', '/mnt/projects/huangwt/wgs/Results-SMuRF/Real-bcbio103-samples/2015-07-31_icgc_cll_T20/snv-allpredictions.txt', '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_cll-N.bam', '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_cll-T20.bam'),

('icgc_mbl-T20', '/home/krishnak/smudl/goldset_files/truth2_positions.txt', '/mnt/projects/huangwt/wgs/Results-SMuRF/Real-bcbio103-samples/2015-07-31_icgc_mbl_T20/snv-allpredictions.txt', '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_mbl-N.bam', '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_mbl-T20.bam'),

('icgc_cll-N30X', '/home/krishnak/smudl/goldset_files/truth1_positions.txt', '/mnt/projects/huangwt/wgs/Results-SMuRF/Real-bcbio103-samples/2015-07-31_icgc_cll_N30/snv-allpredictions.txt', '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_cll-N30.bam', '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_cll-T.bam'),

('icgc_mbl-N30X', '/home/krishnak/smudl/goldset_files/truth2_positions.txt', '/mnt/projects/huangwt/wgs/Results-SMuRF/Real-bcbio103-samples/2015-07-31_icgc_mbl_N30/snv-allpredictions.txt', '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_mbl-N30.bam', '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_mbl-T.bam'),

]

goldset_files_on_nscc = [

('icgc_cll', '/scratch/users/astar/gis/krishnak/smudl/goldset_files/truth1_positions.txt', '/scratch/users/astar/gis/krishnak/SMURF_GOLDSET_PREDICTIONS/icgc_cll', '/scratch/users/astar/gis/krishnak/icgc_cll-N.bam', '/scratch/users/astar/gis/krishnak/icgc_cll-T.bam'),

('icgc_mbl', '/scratch/users/astar/gis/krishnak/smudl/goldset_files/truth2_positions.txt', '/scratch/users/astar/gis/krishnak/SMURF_GOLDSET_PREDICTIONS/icgc_mbl', '/scratch/users/astar/gis/krishnak/icgc_mbl-N.bam', '/scratch/users/astar/gis/krishnak/icgc_mbl-T.bam'),

('icgc_cll-T40', '/scratch/users/astar/gis/krishnak/smudl/goldset_files/truth1_positions.txt', '/scratch/users/astar/gis/krishnak/SMURF_GOLDSET_PREDICTIONS/icgc_cll-T40', '/scratch/users/astar/gis/krishnak/icgc_cll-N.bam', '/scratch/users/astar/gis/krishnak/icgc_cll-T40.bam'),

('icgc_mbl-T40', '/scratch/users/astar/gis/krishnak/smudl/goldset_files/truth2_positions.txt', '/scratch/users/astar/gis/krishnak/SMURF_GOLDSET_PREDICTIONS/icgc_mbl-T40', '/scratch/users/astar/gis/krishnak/icgc_mbl-N.bam', '/scratch/users/astar/gis/krishnak/icgc_mbl-T40.bam'),

('icgc_cll-T20', '/scratch/users/astar/gis/krishnak/smudl/goldset_files/truth1_positions.txt', '/scratch/users/astar/gis/krishnak/SMURF_GOLDSET_PREDICTIONS/icgc_cll-T20', '/scratch/users/astar/gis/krishnak/icgc_cll-N.bam', '/scratch/users/astar/gis/krishnak/icgc_cll-T20.bam'),

('icgc_mbl-T20', '/scratch/users/astar/gis/krishnak/smudl/goldset_files/truth2_positions.txt', '/scratch/users/astar/gis/krishnak/SMURF_GOLDSET_PREDICTIONS/icgc_mbl-T20', '/scratch/users/astar/gis/krishnak/icgc_mbl-N.bam', '/scratch/users/astar/gis/krishnak/icgc_mbl-T20.bam'),

('icgc_cll-N30X', '/scratch/users/astar/gis/krishnak/smudl/goldset_files/truth1_positions.txt', '/scratch/users/astar/gis/krishnak/SMURF_GOLDSET_PREDICTIONS/icgc_cll-N30X', '/scratch/users/astar/gis/krishnak/icgc_cll-N30.bam', '/scratch/users/astar/gis/krishnak/icgc_cll-T.bam'),

('icgc_mbl-N30X', '/scratch/users/astar/gis/krishnak/smudl/goldset_files/truth2_positions.txt', '/scratch/users/astar/gis/krishnak/SMURF_GOLDSET_PREDICTIONS/icgc_mbl-N30X', '/scratch/users/astar/gis/krishnak/icgc_mbl-N30.bam', '/scratch/users/astar/gis/krishnak/icgc_mbl-T.bam')

]

#dream_challenge_files_on_nscc = [ ('synthetic_sample_%d' % x, '/seq/astar/gis/projects/skandera/DREAM_SYNTHETIC_TUMORS/synthetic.challenge.set%d.trues.txt' % x, '/seq/astar/gis/projects/skandera/DREAM_SYNTHETIC_TUMORS/synthetic.challenge.set%d.normal.bam' % x, '/seq/astar/gis/projects/skandera/DREAM_SYNTHETIC_TUMORS/synthetic.challenge.set%d.tumor.bam' % x) for x in range(1, 6) ]

dream_challenge_files_on_nscc = [

('dream%d' % x, '/seq/astar/gis/projects/skandera/training/wgs/dream/dream%d.truth.txt' % x, '/seq/astar/gis/projects/skandera/training/wgs/dream/dream%d-N-ready.bam' % x, '/seq/astar/gis/projects/skandera/training/wgs/dream/dream%d-T-ready.bam' % x) for x in range(1,6)

]

giab_files_on_nscc = [

('giab', '/seq/astar/gis/projects/skandera/training/wgs/giab/giab_truth_snvs.txt', '/seq/astar/gis/projects/skandera/training/wgs/giab/giab-N-ready.bam', '/seq/astar/gis/projects/skandera/training/wgs/giab/giab-T-ready.bam')

]

TGEN_FILES_ON_NSCC = [

('tgen', '/scratch/users/astar/gis/krishnak/TGEN/tgen_snv_trues.txt', '/scratch/users/astar/gis/krishnak/TGEN/tgen_colo829-N-ready.bam', '/scratch/users/astar/gis/krishnak/TGEN/tgen_colo829-T-ready.bam')

]

subsampled_data_folder_name = 'subsampled'
all_data_folder_name = 'all_data'

#### GERMLINE VARIANT FILTER IN POST-PROCESSING VCF ###
GERMLINE_FILTER = True

############# TRAINING PARAMS #########

EPOCHS_TO_TRAIN = 20 # 50 # 300 #10 #  
TRAINING_BATCH_SIZE = 32
VALIDATION_SPLIT = 0.01 #0.2 # percentage of training shuffled files to use for validation (for early stopping) 

TEST_TIME_TRAINING = False
ADVERSARIAL_TRAINING = False
SPECTRAL_DECOUPLING = False # default l2 reg with weight 2e-05
SD_COEFF = 2e-05 # SPECTRAL_DECOUPLING lambda
LR_SCHEDULE = True
LOSS = 'binary_crossentropy' # 'mean_squared_error' # 'categorical_crossentropy' for multi-class
NUM_MODEL_OUTPUTS = 1 # number of output logits of the model

SAMPLE_READS = True # sample reads and sort by CO. don't use this when generating training data if you don't want sampling, or set SAMPLE_READS_COUNT to 1. 
MULTIPLE_READ_SAMPLES = False # if True, create_input_tensor_for_position() will sample multiple sets of reads at each site. If False, samples once. WARNING: This will also affect prediction.
SAMPLE_READS_COUNT = 1 # 20 for training samples WES cohort # 10 for WGS TCGA FFPE cohort # default 1. for use with SAMPLE_READS and MULTIPLE_READ_SAMPLES
ENCODE_READ_ORIENTATION = False # if True, encodes read orientation in snv input in the same channel as strand bias. Not implemented for indels. WARNING: Will affect prediction

INPUT_WHITENING = False # decorrelate input pixels per channel
WHITENING_MATRIX = 'training_data_whitening_matrix.npy' # computed using 10k random training samples. see smudl/utils.py -> compute_whitening_matrix_torch

VARIABLE_INPUT_SIZE = False # if True, all reads at candidate sites are used
INITIALIZE_EXISTING_WEIGHTS = False

STOCHASTIC_WEIGHT_AVERAGING = False # keep an swa copy of the model
EXPONENTIAL_MOVING_AVERAGE = False # keep an ema copy of the model

SPIKE_IN_GOLDSET = False # during training
DOMAIN_ADVERSARIAL_TRAINING = False # DANN
best_combined_dann_model = 'best.combined_model.dann.hdf5'
dann_target_dist_data_folder = 'target_distribution_data'

# TRAINING PARAMS
TUMOR_ONLY_TRAINING = True # False # to train model with tumor-only encoding
BALANCE_CLASSES = False # True # if True, will balance classes during training in train.py
BALANCE_CLASSES_DOWNSAMPLE = False # if True, will downsample majority class

CLASS_0_WEIGHT, CLASS_1_WEIGHT= 1, 1 # class weights for model.fit

PARTIAL_FINETUNING = False # freeze all layers except final layer
REINITIALIZE_LAST_LAYERS = True # use with PARTIAL_FINETUNING. reinitializes layers to finetune without keeping pretrained weights
NUM_LAYERS_TO_FINETUNE = 4 # how many final layers to fine tune 
FINETUNING_LR = 1e-4 # 1e-3 # 0.00001 # default 0.0001
TRAINING_LR = 1e-4 # 1e-4
LABEL_SMOOTHING = 0 # 0.1 # label smoothing used with binary_crossentropy
FINETUNING_WEIGHT_DECAY = 0 # used for finetuning only
FINETUNING_OPTIMIZER = 'adam'
VAL_METRIC = 'F1' # 'AUPRC' # 'F1'

# EXPERIMENTS FOLDER
experiment_name = 'exp_273_convnet2_ffpe_retrain_last_4_layers_lr_1e-2_class_balance' # 'exp_271_convnet2_ffpe_finetune_last_4_layers_lr_1e-3_class_balance_ground_truth_v3' # 'exp_9_transformer_cross_attention_lr_1e-4_attention_dropout_0.1.layer_postprocess_dropout_0.relu_dropout_0' #'exp_181_convnet2_ffpe_finetune_last_2_layers_lr_1e-4_class_balance_ground_truth_v1' # 'convnet2_ffpe_finetune_last_3_layers_lr_1e-4_class_balance_ground_truth_v2' # 'convnet2_ffpe_last_layer_training_lr_0.0001_class_balance_full_finetuning_lr_0.00001' # 'convnet2_ffpe_last_layer_training_lr_0.0001_class_balance' # 'convnet2_tumor_only' # 'convnet2_retrain_subset' # 'convnet2_input_whitening' # 'inceptionv3_train_on_val.ood_val_set' # 'convnet2_train_on_val' # 'inceptionv3_train_on_val' # 'inceptionv3_variable_input_height' # 'snv.inceptionv3.new_architecture.old_weights' # 'convnet2_global_average_pooling' # 'inception' # 'convnet2_mean_squared_error' # 'EMA.convnet2_spectral_decoupling_sd_coeff_%s' % SD_COEFF # 'convnet2_fgsm_rand_l2_eps_0.25' # 'convnet2.lr_0.0001_dr_0.0' #  'convnet2_ttt' # 'convnet2_selu' # 'convnet2_group_normalization' # 'EfficientNetB0'
experiment_name = 'exp_273_convnet2_ffpe_from_scratch_lr_1e-4_class_balance_downsample' 
experiment_name = 'exp_273_convnet2_ffpe_full_finetune_lr_1e-4_class_balance_downsample'
experiment_name = 'convnet2_ffpe_retrain_last_4_layers_lr_1e-3_class_balance_label_smoothing_%s' % str(LABEL_SMOOTHING)
experiment_name = 'TUMOR_ONLY'
experiment_name = 'convnet2_tumor_only_class_weight_0_5_class_weight_1_1' # upweight class 0 by 5

#### TRAINING FOLDERS
training_data_folder_on_aquila = '/mnt/projects/krishnak/kiran/smudl_training_data/'
training_data_folder_on_nscc = '/scratch/users/astar/gis/krishnak/project/smudl_training_data' # '/home/project/13002420/smudl_training_data' #'/scratch/users/astar/gis/krishnak/smudl_training_data'
training_data_folder_on_workstation = '/media/nvme/kiran/smudl_training_data/'
coverage_info_file = 'coverage_info.txt'
mutation_burden_info_file = os.path.join('mutation_burden_info.npy')
training_data_folder = training_data_folder_on_nscc

folder_to_save_trained_models = '/mnt/nvme/smudl_trained_models/'
normalized_training_data_folder_name = 'normalized_training_data'
save_models_to_folder = 'trained_models'
tensorboard_log_directory = 'Tensorboard_logs'

experiment_details_file = 'experiment_details.txt'
patient_names_file = 'list_of_patients_used.npy'
training_filenames = 'list_of_files_used_for_training.npy'
validation_filenames = 'list_files_used_for_validation.npy'
test_filenames = 'list_of_files_used_for_testing.npy'

validation_domain_classifier_accuracy_history_file = 'validation_domain_classifier_accuracy_history.npy'

finetuned_models_dir = 'finetuned_models'

SAVE_EVERY_EPOCH = False # if True, save every epoch, not just the best val acc. model
CHECKPOINT_FREQUENCY = 9999 # 5 

channels_means_file = 'channel_means_for_normalization.npy'
channels_std_devs_file = 'channel_standard_deviations_for_normalization.npy'
shuffled_training_data_folder = 'shuffled_training_data'
shuffled_data_indices = 'shuffled_data_indices.npy'
shuffled_batch_size = 1000 # 10000 # 1000 # 100 # 

experiments_folder_on_aquila = '/mnt/projects/krishnak/kiran/smudl_trained_models/'
experiments_folder_in_nscc =  '/scratch/users/astar/gis/krishnak/project/smudl_trained_models' # '/home/project/13002420/smudl_trained_models'
experiments_folder_in_workstation = '/mnt/nvme/smudl_trained_models/'
experiments_folder = experiments_folder_in_workstation # experiments_folder_in_nscc # experiments_folder_in_workstation # 

# SAMPLE FOLDER NAMES
sample_candidates_folder = 'candidates'
sample_predictions_folder = 'predictions'

snv_candidates_folder = 'snvs'
indel_candidates_folder = 'indels'

""" SNV MODEL """
snv_model_folder = 'snv_model'
NORMALIZATION_MEANS_PATH = os.path.join(snv_model_folder, channels_means_file)
NORMALIZATION_STD_DEVS_PATH = os.path.join(snv_model_folder, channels_std_devs_file)

# name of tumor-normal convnet2 (snv) adapted per sample
SNV_ADAPTED_TUMOR_NORMAL_MODEL = 'adapted_snv_tumor_normal_model.hdf5'

# for tumor-only convnet
TUMOR_ONLY_NORMALIZATION_MEANS_PATH = os.path.join(snv_model_folder, 'tumor_only_channel_means_for_normalization.npy')
TUMOR_ONLY_NORMALIZATION_STD_DEVS_PATH = os.path.join(snv_model_folder, 'tumor_only_channel_standard_deviations_for_normalization.npy')

# # tumor only models (convnet2)
# BEST_TUMOR_ONLY_MODEL_ARCHITECTURE_PATH = os.path.join(snv_model_folder, 'model.best.architecture.tumor_only.json')
# BEST_TUMOR_ONLY_MODEL_WEIGHTS_PATH = os.path.join(snv_model_folder, 'model.best.weights.tumor_only.hdf5')
# BEST_TUMOR_ONLY_MODEL_PATH = os.path.join(snv_model_folder, 'model.best.tumor_only.hdf5')

# tumor-only transformer
BEST_TUMOR_ONLY_MODEL_PATH = 'model.TUMOR_ONLY_TRANSFORMER'

# tumor-only FFPE transformer
BEST_FFPE_TUMOR_ONLY_MODEL_PATH = 'model.FFPE_TUMOR_ONLY_TRANSFORMER'

# normal-tumor models
BEST_MODEL_ARCHITECTURE_PATH = os.path.join(snv_model_folder, 'model.best.architecture.json')
BEST_MODEL_WEIGHTS_PATH = os.path.join(snv_model_folder, 'model.best.weights.hdf5')
BEST_MODEL_PATH = os.path.join(snv_model_folder, 'model.best.hdf5')

# tumor only convnet models (trained on TCGA WXS)
# TUMOR_ONLY_BEST_MODEL_ARCHITECTURE_PATH = os.path.join(snv_model_folder, 'model.best.tumor_only.architecture.json')
# TUMOR_ONLY_BEST_MODEL_WEIGHTS_PATH = os.path.join(snv_model_folder, 'model.best.tumor_only.weights.hdf5')

# exp 9 tumor only (upweight class 0 by 5)
#TUMOR_ONLY_BEST_MODEL_ARCHITECTURE_PATH = os.path.join(snv_model_folder, 'model.best.architecture.convnet2_tumor_only_class_weight_0_5_class_weight_1_1.json')
#TUMOR_ONLY_BEST_MODEL_WEIGHTS_PATH = os.path.join(snv_model_folder, 'model.best.weights.convnet2_tumor_only_class_weight_0_5_class_weight_1_1.hdf5')

# exp 428 (exp 9 model retrained with germline snvs)
# TUMOR_ONLY_BEST_MODEL_ARCHITECTURE_PATH = os.path.join(snv_model_folder, 'model.best.architecture.tumor_only.exp_428_retrain_with_germline.json')
# TUMOR_ONLY_BEST_MODEL_WEIGHTS_PATH = os.path.join(snv_model_folder, 'model.best.weights.tumor_only.exp_428_retrain_with_germline.hdf5')

# exp 428 (exp 9 model retrained with germline snvs) downsample germline 90pct
#TUMOR_ONLY_BEST_MODEL_ARCHITECTURE_PATH = os.path.join(snv_model_folder, 'model.best.architecture.retrain_with_germline_downsample_germline_90pct.json')
#TUMOR_ONLY_BEST_MODEL_WEIGHTS_PATH = os.path.join(snv_model_folder, 'model.best.weights.retrain_with_germline_downsample_germline_90pct.hdf5')

# exp 428 (exp 9 model retrained with germline snvs) train from scratch
#TUMOR_ONLY_BEST_MODEL_ARCHITECTURE_PATH = os.path.join(snv_model_folder, 'model.best.architecture.retrain_with_germline_from_scratch.json')
#TUMOR_ONLY_BEST_MODEL_WEIGHTS_PATH = os.path.join(snv_model_folder, 'model.best.weights.retrain_with_germline_from_scratch.hdf5')

# exp 428 (exp 9 model retrained with germline snvs) train from scratch multi-class
#TUMOR_ONLY_BEST_MODEL_ARCHITECTURE_PATH = os.path.join(snv_model_folder, 'model.best.architecture.retrain_with_germline_from_scratch_multi_class.json')
#TUMOR_ONLY_BEST_MODEL_WEIGHTS_PATH = os.path.join(snv_model_folder, 'model.best.weights.retrain_with_germline_from_scratch_multi_class.hdf5')

# exp 9 tumor only
TUMOR_ONLY_BEST_MODEL_ARCHITECTURE_PATH = os.path.join(snv_model_folder, 'model.best.architecture.tumor_only.exp_9.json')
TUMOR_ONLY_BEST_MODEL_WEIGHTS_PATH = os.path.join(snv_model_folder, 'model.best.weights.tumor_only.exp_9.hdf5')

""" CURRENT EXPERIMENT """
def set_experiment_paths(experiment_id):
	global CURRENT_EXPERIMENT_ID
	CURRENT_EXPERIMENT_ID = experiment_id

	global experiment_name
	experiment_name = 'experiment_%s' % str(CURRENT_EXPERIMENT_ID)

	global CURRENT_EXPERIMENT_FOLDER
	CURRENT_EXPERIMENT_FOLDER = os.path.join(experiments_folder, experiment_name)

	global CURRENT_BEST_MODEL_PATH
	CURRENT_BEST_MODEL_PATH = os.path.join(CURRENT_EXPERIMENT_FOLDER, save_models_to_folder, best_model_name)

	global CURRENT_BEST_MODEL_ARCHITECTURE_PATH
	CURRENT_BEST_MODEL_ARCHITECTURE_PATH = os.path.join(CURRENT_EXPERIMENT_FOLDER, save_models_to_folder, best_model_architecture)

	global CURRENT_BEST_MODEL_WEIGHTS_PATH
	CURRENT_BEST_MODEL_WEIGHTS_PATH = os.path.join(CURRENT_EXPERIMENT_FOLDER, save_models_to_folder, best_model_weights)

	global CURRENT_FINETUNED_MODELS_DIR
	CURRENT_FINETUNED_MODELS_DIR = os.path.join(CURRENT_EXPERIMENT_FOLDER, save_models_to_folder, finetuned_models_dir)

	global CURRENT_NORMALIZATION_MEANS_PATH
	CURRENT_NORMALIZATION_MEANS_PATH = os.path.join(CURRENT_EXPERIMENT_FOLDER, channels_means_file)

	global CURRENT_NORMALIZATION_STD_DEVS_PATH
	CURRENT_NORMALIZATION_STD_DEVS_PATH = os.path.join(CURRENT_EXPERIMENT_FOLDER, channels_std_devs_file)


DEFAULT_EXPERIMENT_ID = 9 #  375 # 375 is FFPE tumor-only transformer # 278 # 273 # 271 # 182 # 9 #  190 # 181 # 180 # 179 # 144 # 85 # 43

### PRE-FILTER POSITIONS IN WHOLE GENOME
filtering_root_folder_on_nscc = '/scratch/users/astar/gis/krishnak/filtered_positions/'
filtering_root_folder_on_aquila = '/home/krishnak/filtered_positions/'
filtering_root_folder = filtering_root_folder_on_aquila
CURRENT_FILTERING_FOLDER = 'filtering_3'

filtering_folder = os.path.join(filtering_root_folder, CURRENT_FILTERING_FOLDER)
filtered_positions_file = 'Positions.csv'
filtering_details_file = 'filter_details.txt'
filtering_batches_folder = os.path.join(filtering_folder, 'output')

# CUSTOM PRE_FILTERS
MIN_BASE_QUALITY = 22
MIN_COVERAGE = 7
MIN_MUTANT_ALLELE_READS_IN_TUMOR = 2
MIN_READ_MAPPING_QUALITY = 10
MAX_ALTERNATIVE_ALLELE_FREQUENCY_IN_NORMAL = 0.05
MIN_MUTANT_ALLELE_FREQUENCY_IN_TUMOR = 0.035
MIN_MAPPING_QUALITY_FOR_MUTANT_ALLELE_READS = 30

# Reference: https://media.nature.com/original/nature-assets/ncomms/2015/151209/ncomms10001/extref/ncomms10001-s1.pdf
# Supplementary table 7
MIN_DISTANCE_FROM_VARIANT_TO_ALIGNMENT_END_MEDIAN = 10
MIN_DISTANCE_FROM_VARIANT_TO_ALIGNMENT_END_MAD = 3 # MAD - Median Absolute Deviation
MAX_PROPORTION_OF_LOW_MAP_QUAL_READS_AT_VARIANT = .10 # low map qual is if MAPQ < 1
MAX_MAP_QUAL_DIFF_MEDIAN = 5 # The difference in the median mapping quality of variant reads (in the tumor) and reference reads (in the normal) is greater than 5
MIN_VARIANT_MAP_QUAL_MEDIAN = 40 # The median mapping quality of variant reads is less than 40
MIN_VARIANT_BASE_QUAL_MEDIAN = 30 # The median base quality at the variant position of variant reads is less than 30
MIN_VARIANT_ALLELE_COUNT = 4 # The number of variant-supporting reads in the tumor is less than 4
MAX_VARIANT_ALLELE_COUNT_IN_CONTROL = 1 # The number of variant-supporting reads in the normal is greater than 1
MIN_STRAND_BIAS = 0.02 # The strand bias for variant reads covering the variant position, i.e. the fraction of reads in either direction, is less than 0.02, unless the strand bias for all reads is also less than 0.02
"""
The largest number of variant positions within any 50 base pair
window surrounding, but excluding, the variant position is greater
than 2; variant positions are those in which the number of
alternate allele is supported by at least 2 reads and at least 5% of
all reads covering that position.
"""
SNVCluster50 = 2
"""
The largest number of variant positions within any 100 base pair
window surrounding, but excluding, the variant position is greater
than 4; variant positions are those in which the number of
alternate allele is supported by at least 2 reads and at least 5% of
all reads covering that position
"""
SNVCluster100 = 2


ref_path_on_workstation = "/mnt/nvme/GRCh37/GRCh37.fa"
predictions_folder_on_workstation = '/home/kiran/smudl_predictions/'

ref_path_on_aquila = "/mnt/projects/huangwt/wgs/genomes/seq/GRCh37.fa"
predictions_folder_on_aquila = '/home/krishnak/smudl_predictions/'

ref_path_on_nscc = "/scratch/users/astar/gis/krishnak/GRCh37.fa"
predictions_folder_on_nscc = '/scratch/users/astar/gis/krishnak/smudl_predictions/'

from os.path import expanduser

HOME_DIRECTORY = expanduser('~')

if HOME_DIRECTORY == '/home/kiran':
	ref_path = ref_path_on_workstation

elif HOME_DIRECTORY == '/home/users/astar/gis/krishnak':
	ref_path = ref_path_on_nscc

elif HOME_DIRECTORY == '/home/krishnak':
	ref_path = ref_path_on_aquila

combined_predictions_file = 'Predictions.csv'

############## INPUT ENCODING SETTINGS #####################

REMOVE_REFERENCE_CHANNEL = False
ADD_NORMAL_TUMOR_FLAG_CHANNEL = False # channel to indicate if position is in normal or tumor

CTR_DUP = 5 # Duplicate the center column, which is to be predicted
SEQ_LENGTH = 31 # must be odd, length of sequence

# SEQ_LENGTH needs to be ODD
assert SEQ_LENGTH % 2

PER_IMAGE_WIDTH = SEQ_LENGTH + CTR_DUP - 1

FLANK = int((SEQ_LENGTH-1)/2)
NUM_READS = 100 # max number of reads to include / array height

MAX_READS = 500 # max reads to use if all reads are used during prediction (when VARIABLE_INPUT_SIZE=True)

# tumor and normal images adjacent in the encoding. If False, tumor and normal stacked one behind another
TUMOR_NORMAL_ADJACENT = True

TETRIS_MODE = False # if False, only one read is encoded in each row of the image. If true, reads fill up the image from the top down per position, even if reads have to be broken
ENCODE_INSERTIONS = False
SORT_BASES = False # sort each column by base A, T, G, C

SMALLER_INPUT = True

def set_input_encoding(af):
	global NUM_CHANNELS_PER_IMAGE
	NUM_CHANNELS_PER_IMAGE = 4

	if af:
		NUM_CHANNELS_PER_IMAGE += 4 # one channel for each base

	global INCLUDE_ALLELE_FREQUENCY
	INCLUDE_ALLELE_FREQUENCY = af

	global NUM_CHANNELS
	global INPUT_SHAPE
	global TUMOR_ONLY_INPUT_SHAPE
	
	if TUMOR_NORMAL_ADJACENT:
		if REMOVE_REFERENCE_CHANNEL:
			NUM_CHANNELS = NUM_CHANNELS_PER_IMAGE
		else:
			NUM_CHANNELS = NUM_CHANNELS_PER_IMAGE + 1 # add one for reference channel

		if ADD_NORMAL_TUMOR_FLAG_CHANNEL:
			NUM_CHANNELS += 1

		INPUT_SHAPE = [ NUM_READS, 2*PER_IMAGE_WIDTH, NUM_CHANNELS]
		TUMOR_ONLY_INPUT_SHAPE = [ NUM_READS, PER_IMAGE_WIDTH, NUM_CHANNELS]
	else:
		NUM_CHANNELS = NUM_CHANNELS_PER_IMAGE + NUM_CHANNELS_PER_IMAGE + 1 # tumor + normal + 1 for ref channel
		INPUT_SHAPE = [ NUM_READS, SEQ_LENGTH + CTR_DUP - 1, NUM_CHANNELS]

DEFAULT_INCLUDE_ALLELE_FREQUENCY = False
set_input_encoding(DEFAULT_INCLUDE_ALLELE_FREQUENCY)

# if True, compares bases to the reference base. Encodes 0 if they are the same
COMPARE_REF_BASE = False

def set_encoding(tetris_mode, encode_insertions, sort_bases):
	global encoding_name
	encoding_name = 'NUM_READS_%s_SEQ_LENGTH_%s_CTR_DUP_%s_NUM_CHANNELS_%s_NUM_CHANNELS_PER_IMG_%s' % (str(NUM_READS), str(SEQ_LENGTH), str(CTR_DUP), str(NUM_CHANNELS), str(NUM_CHANNELS_PER_IMAGE) )

	if tetris_mode:
		encoding_name += '_TETRIS_MODE'

	if encode_insertions:
		encoding_name += '_ENCODE_INSERTIONS'

	if sort_bases:
		encoding_name += '_SORT_BASES'

set_encoding(TETRIS_MODE, ENCODE_INSERTIONS, SORT_BASES)

## TUMOR ONLY settings for transformer 
TRAINING_TUMOR_ONLY_TRANSFORMER = False # True # frozen tumor-only transformer
TRAINING_TUMOR_ONLY_FFPE_TRANSFORMER = False # varnet-ffpe tumor-only transformer

TUMOR_ONLY_FLANK = 20 # 5, 10, 20, 30, 40, 50 #  Flank length on left and right of candidate position
TUMOR_ONLY_HEIGHT = 108 # number of rows
TUMOR_ONLY_SHAPE = (TUMOR_ONLY_HEIGHT, 2*TUMOR_ONLY_FLANK + 1) # candidate position column is encoded once only (no duplication)

TUMOR_ONLY_ENCODING_NAME = 'TUMOR_ONLY_FLANK_%d_HEIGHT_%d' % (TUMOR_ONLY_FLANK, TUMOR_ONLY_HEIGHT)
# TUMOR_ONLY_ENCODING_NAME = 'TUMOR_ONLY_FLANK_%d_HEIGHT_%d_PASS_CALL_ARTIFACTS' % (TUMOR_ONLY_FLANK, TUMOR_ONLY_HEIGHT) # pass calls from callers not in ground-truth are used as artifacts

def set_experiment_name(experiment_name):
	global validation_f1_score_history_file
	validation_f1_score_history_file = 'validation_f1_score_history.%s.npy' % experiment_name
	global validation_precision_history_file
	validation_precision_history_file = 'validation_precision_history.%s.npy' % experiment_name
	global validation_recall_history_file
	validation_recall_history_file = 'validation_recall_history.%s.npy' % experiment_name

	global test_f1_score_history_file
	test_f1_score_history_file = 'test_f1_score_history.%s.npy' % experiment_name

	global config_file
	config_file = 'config_%s.npy' % experiment_name # save constants file as dictionary
	
	global best_model_name
	
	if TRAINING_TUMOR_ONLY_FFPE_TRANSFORMER:
		best_model_name = 'model.best.%s_TRANSFORMER' % experiment_name # directory name, shouldn't end in .hdf5
	else:
		best_model_name = 'model.best.%s.hdf5' % experiment_name

	global initial_model_name
	initial_model_name = 'dummy' # 'model.best.lr_0.0001_dr_0.0.hdf5' # 'dummy' # 'model.best.exp_182_convnet2_ffpe_finetune_last_4_layers_lr_1e-3_class_balance_ground_truth_v3.hdf5' # 'model.initial.%s.hdf5' % architecture # initial model to continue training
	global best_model_architecture
	best_model_architecture = 'model.best.architecture.%s.json' % experiment_name # architecture only of the best model as a json file
	global best_model_weights
	best_model_weights = 'model.best.weights.%s.hdf5' % experiment_name # weights only
	global SWA_WEIGHTS_MODEL
	SWA_WEIGHTS_MODEL = 'model.SWA.%s.hdf5' % experiment_name
	global EMA_WEIGHTS_MODEL
	EMA_WEIGHTS_MODEL = 'model.EMA.%s.hdf5' % experiment_name

set_experiment_name(experiment_name)

__VERSION__ = '1.5.1'
