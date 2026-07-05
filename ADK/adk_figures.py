# -*- coding: utf-8 -*-
"""
process_adk_all_figures_original_window_svg_folder.py

Combined ADK script.

Part 1 uses the uploaded original `Ca - window.py` window-selection code
directly, except that Fig. 4(b) is saved as SVG into OUTPUT_DIR.

Part 2 uses the renamed-only `Ca.py` figure-generation code, except that
Fig. 4(c), Fig. 4(d), and Fig. S3 are saved as SVG into OUTPUT_DIR.

No window value, residue selection, SVD logic, noise setting, or plotting
layout is intentionally changed.
"""

import os 
import shutil 

OUTPUT_DIR ="adk_results"


def reset_output_dir ():
    """Create a clean SVG-only output folder."""
    if os .path .isdir (OUTPUT_DIR ):
        shutil .rmtree (OUTPUT_DIR )
    os .makedirs (OUTPUT_DIR ,exist_ok =True )


reset_output_dir ()


# ============================================================
# Part 1: original Ca - window.py logic for Fig. 4(b)
# ============================================================

import mdtraj as md 
import numpy as np 
import matplotlib .pyplot as plt 
from scipy .signal import find_peaks 
from matplotlib .ticker import MaxNLocator 
import warnings 
warnings .filterwarnings ("ignore")

plt .rcParams ["font.family"]="Arial"
plt .rcParams ["mathtext.fontset"]="custom"
plt .rcParams ["mathtext.rm"]="Arial"
plt .rcParams ["mathtext.it"]="Arial:italic"
plt .rcParams ["mathtext.bf"]="Arial:bold"
plt .rcParams ["axes.unicode_minus"]=False 
plt .rcParams ["svg.fonttype"]="none"

PDB_FILE ="closedCA.pdb"
TRAJECTORY_FILE ="C1.0_1.dcd"

conf =md .load (PDB_FILE ,top =PDB_FILE )
natoms =conf .n_atoms 
trajectory =md .load (TRAJECTORY_FILE ,top =PDB_FILE ,atom_indices =range (natoms ))
ref_frame =trajectory [0 ]
aligned_trajectory =trajectory .superpose (ref_frame )

time_grid_t =aligned_trajectory .time 
coordinate_tensor_xyz =aligned_trajectory .xyz 
coordinate_tensor_atom_xyz_time =np .transpose (coordinate_tensor_xyz ,(1 ,2 ,0 ))
flattened_coordinate_matrix =coordinate_tensor_atom_xyz_time .reshape (-1 ,coordinate_tensor_atom_xyz_time .shape [-1 ])
print ("flattened_coordinates.shape =",flattened_coordinate_matrix .shape )

ANGLE_RESIDUE_GROUPS_NMP ={"CORE":(90 ,100 ),"CORE-LID":(115 ,125 ),"NMP":(35 ,55 )}
ANGLE_RESIDUE_GROUPS_LID ={"CORE-LID":(179 ,185 ),"CORE":(115 ,125 ),"NMP":(125 ,153 )}

def extract_coordinates (aligned_trajectory ,residue_ranges ):
    coordinates_dict ={}
    for name ,(start ,end )in residue_ranges .items ():
        atom_indices =list (range (start ,end +1 ))
        coordinates_dict [name ]=aligned_trajectory .atom_slice (atom_indices ).xyz 
    return coordinates_dict 

def calculate_geometric_centers_over_time (coordinates ):
    centers_over_time ={}
    for name ,coords in coordinates .items ():
        centers_over_time [name ]=np .mean (coords ,axis =1 )
    return centers_over_time 

def construct_vectors (centers_over_time ):
    core_centers =centers_over_time ["CORE"]
    core_lid_centers =centers_over_time ["CORE-LID"]
    lid_centers =centers_over_time ["NMP"]
    v1 =core_lid_centers -core_centers 
    v2 =lid_centers -core_centers 
    return v1 ,v2 

def calculate_angle_between_vectors (v1 ,v2 ):
    dot_product =np .sum (v1 *v2 ,axis =1 )
    norm_v1 =np .linalg .norm (v1 ,axis =1 )
    norm_v2 =np .linalg .norm (v2 ,axis =1 )
    cos_theta =dot_product /(norm_v1 *norm_v2 )
    angle =np .arccos (np .clip (cos_theta ,-1.0 ,1.0 ))*(180 /np .pi )
    return angle 

coordinates_theta_NMP =extract_coordinates (aligned_trajectory ,ANGLE_RESIDUE_GROUPS_NMP )
centers_theta_NMP =calculate_geometric_centers_over_time (coordinates_theta_NMP )
vector1_theta_NMP ,vector2_theta_NMP =construct_vectors (centers_theta_NMP )
angle_theta_NMP_degrees =calculate_angle_between_vectors (vector1_theta_NMP ,vector2_theta_NMP )
angle_theta_NMP =np .radians (angle_theta_NMP_degrees )

coordinates_theta_LID =extract_coordinates (aligned_trajectory ,ANGLE_RESIDUE_GROUPS_LID )
centers_theta_LID =calculate_geometric_centers_over_time (coordinates_theta_LID )
vector1_theta_LID ,vector2_theta_LID =construct_vectors (centers_theta_LID )
angle_theta_LID_degrees =calculate_angle_between_vectors (vector1_theta_LID ,vector2_theta_LID )
angle_theta_LID =np .radians (angle_theta_LID_degrees )

selected_coordinate_row_ranges =[(0 ,642 )]
observed_coordinate_matrix_parts =[flattened_coordinate_matrix [start :end ,:]for start ,end in selected_coordinate_row_ranges ]
n_columns_by_selected_part =[part .shape [1 ]for part in observed_coordinate_matrix_parts ]
if len (set (n_columns_by_selected_part ))!=1 :
    raise ValueError ("All parts must have the same number of columns.")

observed_coordinate_matrix =np .vstack (observed_coordinate_matrix_parts )
observed_coordinate_matrix =observed_coordinate_matrix [:,50000 :70000 ]
print ("mar.shape (used slice) =",observed_coordinate_matrix .shape )

NOISE_STD_COORDINATE =0.2 
coordinate_noise =np .random .normal (0 ,NOISE_STD_COORDINATE ,observed_coordinate_matrix .shape )
observed_coordinate_matrix_noisy =observed_coordinate_matrix 

OBSERVABLE_WINDOW_LENGTH =20000 
DELAY_TAU =1 

observable_matrix_blocks_H =[]
for i in range (observed_coordinate_matrix_noisy .shape [0 ]):
    row_data =observed_coordinate_matrix_noisy [i ,:]
    num_rows =len (row_data )-DELAY_TAU *(OBSERVABLE_WINDOW_LENGTH -1 )
    num_cols =DELAY_TAU *OBSERVABLE_WINDOW_LENGTH 
    if num_rows <=0 :
        raise ValueError (f"embedding_dimension={OBSERVABLE_WINDOW_LENGTH } 太大，导致 num_rows={num_rows } <= 0。请减小 embedding_dimension。")
    observable_matrix_block_H =np .zeros ((num_rows ,num_cols ))
    for k in range (num_rows ):
        for l in range (OBSERVABLE_WINDOW_LENGTH ):
            observable_matrix_block_H [k ,l *DELAY_TAU :(l +1 )*DELAY_TAU ]=row_data [k +l *DELAY_TAU ]
    observable_matrix_blocks_H .append (observable_matrix_block_H )

observable_matrix_H =np .vstack (observable_matrix_blocks_H )
print ("Hankel matrix shape:",observable_matrix_H .shape )

left_singular_vectors_U ,singular_values ,right_singular_vectors_Vh =np .linalg .svd (observable_matrix_H ,full_matrices =False )
singular_values_squared_sum =np .sum (singular_values **2 )
singular_values_proportions =(singular_values **2 )/singular_values_squared_sum 
cumulative_energy =np .cumsum (singular_values_proportions )
SVD_ENERGY_THRESHOLD =0.99 
retained_rank_r =np .argmax (cumulative_energy >=SVD_ENERGY_THRESHOLD )+1 
print (f"Selected first {retained_rank_r } singular values, cumulative energy: {cumulative_energy [retained_rank_r -1 ]:.4f}")

reduced_coordinates_V =np .diag (singular_values [:retained_rank_r ])@right_singular_vectors_Vh [:retained_rank_r ,:]

REDUCED_TIME_START_INDEX =50000 
REDUCED_TIME_LENGTH =reduced_coordinates_V .shape [1 ]
reduced_time_grid_t =time_grid_t [REDUCED_TIME_START_INDEX :REDUCED_TIME_START_INDEX +REDUCED_TIME_LENGTH ]
if len (reduced_time_grid_t )<REDUCED_TIME_LENGTH :
    estimated_delta_t =time_grid_t [1 ]-time_grid_t [0 ]
    reduced_time_grid_t =np .linspace (time_grid_t [REDUCED_TIME_START_INDEX ],time_grid_t [REDUCED_TIME_START_INDEX ]+estimated_delta_t *(REDUCED_TIME_LENGTH -1 ),REDUCED_TIME_LENGTH )

reduced_velocity_V_dot =np .zeros_like (reduced_coordinates_V )
for i in range (retained_rank_r ):
    reduced_velocity_V_dot [i ,:]=np .gradient (reduced_coordinates_V [i ,:],reduced_time_grid_t [1 ]-reduced_time_grid_t [0 ],edge_order =2 )

NOISE_STD_V_DOT =0.1 
reduced_velocity_V_dot +=np .random .normal (0 ,NOISE_STD_V_DOT ,reduced_velocity_V_dot .shape )
print ("v.shape, dv_dt.shape =",reduced_coordinates_V .shape ,reduced_velocity_V_dot .shape )

def find_peak_interval_for_window_T (signal ,prominence_rel =0.15 ,distance =10 ,fallback_window_T =150 ):
    if np .all (np .isnan (signal )):
        return fallback_window_T ,np .array ([],dtype =int )
    prom =prominence_rel *(np .nanmax (signal )-np .nanmin (signal ))
    if prom <=0 :
        prom =np .nanstd (signal )*0.1 
    peak_indices ,_ =find_peaks (signal ,prominence =prom ,distance =distance )
    if len (peak_indices )>=2 :
        peak_intervals =np .diff (peak_indices )
        selected_window_T =int (np .max (peak_intervals ))
        return selected_window_T ,peak_indices 
    return fallback_window_T ,peak_indices 

def compute_closure_defect_beta_from_V (reduced_coordinates_V_matrix ,reduced_velocity_V_dot_matrix ,window_length_T ):
    L =reduced_coordinates_V_matrix .shape [1 ]
    window_length_T =min (window_length_T ,L -2 )
    closure_defect_beta_values =[]
    for i in range (0 ,L -window_length_T ):
        V_t =reduced_coordinates_V_matrix [:,i :i +window_length_T ]
        V_dot_t =reduced_velocity_V_dot_matrix [:,i :i +window_length_T ]
        try :
            A_t =V_dot_t @np .linalg .pinv (V_t )
            R_t =V_dot_t -A_t @V_t 
            R_t_row_norms =np .linalg .norm (R_t ,axis =1 )
            closure_defect_beta_values .append (np .mean (R_t_row_norms ))
        except np .linalg .LinAlgError :
            closure_defect_beta_values .append (np .nan )
    return np .array (closure_defect_beta_values )

print ("\n=== Two iterations (ADK / singular-coordinate-signal driven) ===")

first_layer_signal_for_peak_detection =reduced_coordinates_V [0 ,:]
WINDOW_T1 ,first_layer_peak_indices =find_peak_interval_for_window_T (first_layer_signal_for_peak_detection ,prominence_rel =0.15 ,distance =10 ,fallback_window_T =150 )
print ("First suggested window WINDOW_T1 =",WINDOW_T1 ,"detected main peak count:",len (first_layer_peak_indices ))

closure_defect_beta1 =compute_closure_defect_beta_from_V (reduced_coordinates_V ,reduced_velocity_V_dot ,WINDOW_T1 )
closure_defect_beta1_mean =np .nanmean (closure_defect_beta1 )
closure_defect_beta1_variance =np .nanvar (closure_defect_beta1 )
closure_defect_beta1_variance_over_mean =closure_defect_beta1_variance /(closure_defect_beta1_mean +1e-12 )
print (f"First nonlinear term: mean={closure_defect_beta1_mean :.6e}, var={closure_defect_beta1_variance :.6e}, var/mean={closure_defect_beta1_variance_over_mean :.6e}")

WINDOW_T2 ,second_layer_peak_indices =find_peak_interval_for_window_T (closure_defect_beta1 ,prominence_rel =0.1 ,distance =5 ,fallback_window_T =WINDOW_T1 )
print ("Second suggested window WINDOW_T2 =",WINDOW_T2 ,"detected main peak count:",len (second_layer_peak_indices ))

closure_defect_beta2 =compute_closure_defect_beta_from_V (reduced_coordinates_V ,reduced_velocity_V_dot ,WINDOW_T2 )
closure_defect_beta2_mean =np .nanmean (closure_defect_beta2 )
closure_defect_beta2_variance =np .nanvar (closure_defect_beta2 )
closure_defect_beta2_variance_over_mean =closure_defect_beta2_variance /(closure_defect_beta2_mean +1e-12 )
print (f"Second nonlinear term: mean={closure_defect_beta2_mean :.6e}, var={closure_defect_beta2_variance :.6e}, var/mean={closure_defect_beta2_variance_over_mean :.6e}")

improvement_pct =0.0 
if closure_defect_beta1_variance_over_mean !=0 :
    improvement_pct =(closure_defect_beta1_variance_over_mean -closure_defect_beta2_variance_over_mean )/(closure_defect_beta1_variance_over_mean +1e-12 )*100.0 

def _interval_count_distribution (peak_indices ):
    if peak_indices is None or len (peak_indices )<2 :
        return np .array ([],dtype =int ),np .array ([],dtype =int ),np .array ([],dtype =int )
    peak_intervals =np .diff (np .asarray (peak_indices )).astype (int )
    interval_values ,interval_counts =np .unique (peak_intervals ,return_counts =True )
    return peak_intervals ,interval_values ,interval_counts 

def _paper_axes (ax ):
    ax .spines ["top"].set_visible (False )
    ax .spines ["right"].set_visible (False )
    ax .spines ["left"].set_linewidth (1.0 )
    ax .spines ["bottom"].set_linewidth (1.0 )
    ax .tick_params (direction ="out",length =3.5 ,width =1.0 ,pad =3 )

def plot_two_layer_interval_counts_one_figure_ADK (first_layer_peak_indices ,WINDOW_T1 ,second_layer_peak_indices ,WINDOW_T2 ,save_name ="fig4b"):
    first_layer_peak_intervals ,first_layer_interval_values ,first_layer_interval_counts =_interval_count_distribution (first_layer_peak_indices )
    second_layer_peak_intervals ,second_layer_interval_values ,second_layer_interval_counts =_interval_count_distribution (second_layer_peak_indices )

    FIG_W ,FIG_H =10 ,3 
    DPI_SHOW =300 
    LEFT1 =0.105 
    LEFT2 =0.569 
    BOTTOM =0.23 
    AX_W =0.320 
    AX_H =0.68 
    LABEL_SIZE ,TICK_SIZE =20 ,20 
    BAR_COLOR ="k"
    XLABEL_Y =0.075 

    fig =plt .figure (figsize =(FIG_W ,FIG_H ),dpi =DPI_SHOW ,facecolor ="white")
    fig .patch .set_facecolor ("white")

    ax1 =fig .add_axes ([LEFT1 ,BOTTOM ,AX_W ,AX_H ],facecolor ="white")
    ax2 =fig .add_axes ([LEFT2 ,BOTTOM ,AX_W ,AX_H ],facecolor ="white")

    if first_layer_interval_values .size >0 :
        ax1 .bar (first_layer_interval_values ,first_layer_interval_counts ,width =0.9 ,color =BAR_COLOR ,edgecolor =BAR_COLOR ,linewidth =0.3 ,alpha =1.0 )
        ax1 .axvline (WINDOW_T1 ,linestyle ="--",linewidth =1.1 ,color ="k")
        ymax1 =max (int (first_layer_interval_counts .max ()),1 )
        ax1 .set_ylim (0 ,ymax1 *1.20 )
        ax1 .set_ylabel ("Count",fontsize =LABEL_SIZE ,labelpad =8 )
        ax1 .tick_params (axis ="both",labelsize =TICK_SIZE ,labelcolor ="black")
        ax1 .xaxis .set_major_locator (MaxNLocator (nbins =4 ))
        ax1 .yaxis .set_major_locator (MaxNLocator (nbins =3 ))
    else :
        ax1 .text (0.5 ,0.5 ,"Not enough peaks",ha ="center",va ="center",fontsize =LABEL_SIZE ,transform =ax1 .transAxes )
        ax1 .set_axis_off ()

    if second_layer_interval_values .size >0 :
        ax2 .bar (second_layer_interval_values ,second_layer_interval_counts ,width =0.9 ,color =BAR_COLOR ,edgecolor =BAR_COLOR ,linewidth =0.3 ,alpha =1.0 )
        ax2 .axvline (WINDOW_T2 ,linestyle ="--",linewidth =1.1 ,color ="k")
        ymax2 =max (int (second_layer_interval_counts .max ()),1 )
        ax2 .set_ylim (0 ,ymax2 *1.20 )
        ax2 .set_ylabel ("Count",fontsize =LABEL_SIZE ,labelpad =8 )
        ax2 .tick_params (axis ="both",labelsize =TICK_SIZE ,labelcolor ="black")
        ax2 .xaxis .set_major_locator (MaxNLocator (nbins =4 ))
        ax2 .yaxis .set_major_locator (MaxNLocator (nbins =3 ))
    else :
        ax2 .text (0.5 ,0.5 ,"Not enough peaks",ha ="center",va ="center",fontsize =LABEL_SIZE ,transform =ax2 .transAxes )
        ax2 .set_axis_off ()

    for ax in [ax1 ,ax2 ]:
        _paper_axes (ax )
        ax .grid (False )

    fig .text (0.5 ,XLABEL_Y ,r"Peak interval $\Delta k$ (samples)",ha ="center",va ="center",fontsize =LABEL_SIZE )
    output_path =os .path .join (OUTPUT_DIR ,f"{save_name }.svg")
    fig .savefig (output_path ,format ="svg",dpi =DPI_SHOW ,facecolor ="white",transparent =False )
    print (f"Saved: {output_path }")
    plt .show (block =True )

    print ("\n[ADK Interval statistics]")
    if first_layer_peak_intervals .size >0 :
        print (f"Layer-1: min={first_layer_peak_intervals .min ()}, median={int (np .median (first_layer_peak_intervals ))}, max(W1)={first_layer_peak_intervals .max ()}")
    else :
        print ("Layer-1: N/A")
    if second_layer_peak_intervals .size >0 :
        print (f"Layer-2: min={second_layer_peak_intervals .min ()}, median={int (np .median (second_layer_peak_intervals ))}, max(W2)={second_layer_peak_intervals .max ()}")
    else :
        print ("Layer-2: N/A")

plot_two_layer_interval_counts_one_figure_ADK (first_layer_peak_indices =first_layer_peak_indices ,WINDOW_T1 =WINDOW_T1 ,second_layer_peak_indices =second_layer_peak_indices ,WINDOW_T2 =WINDOW_T2 ,save_name ="fig4b")

print ("\n=== Iteration summary (ADK) ===")
print (f"First iteration - window: {WINDOW_T1 }, nonlinear mean: {closure_defect_beta1_mean :.6e}, variance: {closure_defect_beta1_variance :.6e}, variance/mean: {closure_defect_beta1_variance_over_mean :.6e}")
print (f"Second iteration - window: {WINDOW_T2 }, nonlinear mean: {closure_defect_beta2_mean :.6e}, variance: {closure_defect_beta2_variance :.6e}, variance/mean: {closure_defect_beta2_variance_over_mean :.6e}")
print (f"Improvement percentage in variance/mean: {improvement_pct :.2f}%")
print (f"\nNumber of main peaks in the singular-coordinate signal: {len (first_layer_peak_indices )}")
print (f"Number of peaks in the first nonlinear term used for the second iteration: {len (second_layer_peak_indices )}")

# ============================================================
# Part 2: original Ca.py logic for Fig. 4(d), Fig. S3, and Fig. 4(c)
# ============================================================

import mdtraj as md 
import numpy as np 
import matplotlib .pyplot as plt 
from matplotlib .ticker import MaxNLocator 

# =========================================================
# 0. 全局绘图风格：和 window 图统一
# =========================================================
plt .rcParams .update ({
"font.family":"Arial",
"mathtext.fontset":"custom",
"mathtext.rm":"Arial",
"mathtext.it":"Arial:italic",
"mathtext.bf":"Arial:bold",
"axes.unicode_minus":False ,
"svg.fonttype":"none",
})

FIG_W =10 
FIG_H_FIG4D =6.0 
FIG_H_FIGS3 =3.0 
DPI_SHOW =300 

# 这一组参数与 window 图一致
LEFT1 =0.105 
LEFT2 =0.569 
BOTTOM =0.23 
AX_W =0.320 
AX_H =0.68 
XLABEL_Y =0.075 

# beta 图横向范围：从 window 左图左边界，到 window 右图右边界
BETA_LEFT =LEFT1 
BETA_RIGHT =LEFT2 +AX_W 
BETA_WIDTH =BETA_RIGHT -BETA_LEFT 

LABEL_SIZE =20 
TICK_SIZE =20 
LINE_WIDTH =1.2 
SPINE_WIDTH =1.0 

PANEL_BOTTOM =0.125 
PANEL_TOP =0.960 
XLABEL_Y_COMMON =0.055 
# =========================================================
# 1. 数据读取与轨迹对齐
# =========================================================
PDB_FILE ='closedCA.pdb'
TRAJECTORY_FILE ='C1.0_1.dcd'

conf =md .load (PDB_FILE ,top =PDB_FILE )
atom_names =[atom .name for atom in conf .top .atoms ]
natoms =conf .n_atoms 

trajectory =md .load (TRAJECTORY_FILE ,top =PDB_FILE ,atom_indices =range (natoms ))
ref_frame =trajectory [0 ]
aligned_trajectory =trajectory .superpose (ref_frame )

time_grid_t =aligned_trajectory .time 
coordinate_tensor_xyz =aligned_trajectory .xyz 

coordinate_tensor_atom_xyz_time =np .transpose (coordinate_tensor_xyz ,(1 ,2 ,0 ))
flattened_coordinate_matrix =coordinate_tensor_atom_xyz_time .reshape (-1 ,coordinate_tensor_atom_xyz_time .shape [-1 ])

print ("flattened_coordinates shape:",flattened_coordinate_matrix .shape )

# =========================================================
# 2. theta 角度计算部分：保留原代码
# =========================================================
ANGLE_RESIDUE_GROUPS_NMP ={
'CORE':(90 ,100 ),
'CORE-LID':(115 ,125 ),
'NMP':(35 ,55 )
}

ANGLE_RESIDUE_GROUPS_LID ={
'CORE-LID':(179 ,185 ),
'CORE':(115 ,125 ),
'NMP':(125 ,153 )
}


def extract_coordinates (aligned_trajectory ,residue_ranges ):
    coordinates_dict ={}

    for name ,(start ,end )in residue_ranges .items ():
        atom_indices =list (range (start ,end +1 ))
        coordinates =aligned_trajectory .atom_slice (atom_indices ).xyz 
        coordinates_dict [name ]=coordinates 

    return coordinates_dict 


def calculate_geometric_centers (coordinates ):
    geometric_centers ={}

    for name ,coords in coordinates .items ():
        geometric_centers [name ]=np .mean (coords ,axis =(0 ,1 ))

    return geometric_centers 


def calculate_geometric_centers_over_time (coordinates ):
    centers_over_time ={}

    for name ,coords in coordinates .items ():
        centers_over_time [name ]=np .mean (coords ,axis =1 )

    return centers_over_time 


def construct_vectors (centers_over_time ):
    core_centers =centers_over_time ['CORE']
    core_lid_centers =centers_over_time ['CORE-LID']
    lid_centers =centers_over_time ['NMP']

    v1 =core_lid_centers -core_centers 
    v2 =lid_centers -core_centers 

    return v1 ,v2 


def calculate_angle_between_vectors (v1 ,v2 ):
    dot_product =np .sum (v1 *v2 ,axis =1 )
    norm_v1 =np .linalg .norm (v1 ,axis =1 )
    norm_v2 =np .linalg .norm (v2 ,axis =1 )

    cos_theta =dot_product /(norm_v1 *norm_v2 )
    angle =np .arccos (np .clip (cos_theta ,-1.0 ,1.0 ))*(180 /np .pi )

    return angle 


coordinates_theta_NMP =extract_coordinates (aligned_trajectory ,ANGLE_RESIDUE_GROUPS_NMP )
centers_theta_NMP =calculate_geometric_centers_over_time (coordinates_theta_NMP )
vector1_theta_NMP ,vector2_theta_NMP =construct_vectors (centers_theta_NMP )
angle_theta_NMP_degrees =calculate_angle_between_vectors (vector1_theta_NMP ,vector2_theta_NMP )
angle_theta_NMP =np .radians (angle_theta_NMP_degrees )

coordinates_theta_LID =extract_coordinates (aligned_trajectory ,ANGLE_RESIDUE_GROUPS_LID )
centers_theta_LID =calculate_geometric_centers_over_time (coordinates_theta_LID )
vector1_theta_LID ,vector2_theta_LID =construct_vectors (centers_theta_LID )
angle_theta_LID_degrees =calculate_angle_between_vectors (vector1_theta_LID ,vector2_theta_LID )
angle_theta_LID =np .radians (angle_theta_LID_degrees )

angle_matrix_theta =np .column_stack ((angle_theta_NMP ,angle_theta_LID ))
angle_matrix_theta =angle_matrix_theta .T 

# =========================================================
# 3. ADK / Hankel 参数
# =========================================================
ANALYSIS_START_INDEX =50000 
ANALYSIS_END_INDEX =70000 

OBSERVABLE_WINDOW_LENGTH =20000 
DELAY_TAU =1 
RETAINED_RANK_R =3 
WINDOW_T =885 

# 噪声参数：fig4d 不加噪声，figs3 加噪声
NOISE_STD =0.2 
RANDOM_SEED =1 

# 三种区域选择
# 注意：Python 切片右端不包含
OBSERVED_REGION_CASES =[
{
"name":"CORE+NMP+LID",
"rows":[(0 ,60 ),(90 ,150 ),(366 ,426 )],
"color":"#C42238",
"label":r'$\beta$ [1-20(CORE), 30-50(NMP), 122-142(LID)]'
},
{
"name":"CORE+NMP",
"rows":[(0 ,60 ),(90 ,150 )],
"color":"#2A22C4",
"label":r'$\beta$ [1-20(CORE), 30-50(NMP)]'
},
{
"name":"CORE",
"rows":[(0 ,60 )],
"color":"#22C442",
"label":r'$\beta$ [1-20(CORE)]'
}
]

# =========================================================
# 4. 构造 Hankel 矩阵
# =========================================================
def build_observable_matrix_H (observed_coordinate_matrix ,OBSERVABLE_WINDOW_LENGTH ,DELAY_TAU ):
    observable_matrix_blocks_H =[]

    for i in range (observed_coordinate_matrix .shape [0 ]):
        row_data =observed_coordinate_matrix [i ,:]
        num_rows =len (row_data )-DELAY_TAU *(OBSERVABLE_WINDOW_LENGTH -1 )

        if num_rows <=0 :
            raise ValueError ("embedding_dimension 太大，当前数据长度不够构造 Hankel 矩阵。")

        lag_index =np .arange (OBSERVABLE_WINDOW_LENGTH )*DELAY_TAU 
        window_index =np .arange (num_rows )[:,None ]+lag_index [None ,:]
        observable_matrix_block_H =row_data [window_index ]

        observable_matrix_blocks_H .append (observable_matrix_block_H )

    observable_matrix_H =np .vstack (observable_matrix_blocks_H )

    return observable_matrix_H 

    # =========================================================
    # 5. 给定区域，计算 beta
    # =========================================================
def compute_closure_defect_beta_from_rows (
flattened_coordinate_matrix ,
time_grid_t ,
selected_coordinate_row_ranges ,
data_start ,
data_end ,
OBSERVABLE_WINDOW_LENGTH ,
DELAY_TAU ,
RETAINED_RANK_R ,
WINDOW_T ,
add_noise =False ,
NOISE_STD_COORDINATE =0.2 ,
random_seed =None 
):
    observed_coordinate_matrix_parts =[
    flattened_coordinate_matrix [start :end ,:]
    for start ,end in selected_coordinate_row_ranges 
    ]

    n_columns_by_selected_part =[part .shape [1 ]for part in observed_coordinate_matrix_parts ]

    if len (set (n_columns_by_selected_part ))!=1 :
        raise ValueError ("All selected parts must have the same number of columns.")

    observed_coordinate_matrix =np .vstack (observed_coordinate_matrix_parts )
    observed_coordinate_matrix =observed_coordinate_matrix [:,data_start :data_end ]

    # =====================================================
    # 噪声部分
    # fig4d: add_noise=False，不加噪声
    # figs3: add_noise=True，加噪声
    # =====================================================
    if add_noise :
        if random_seed is not None :
            np .random .seed (random_seed )

        coordinate_noise =np .random .normal (0 ,NOISE_STD_COORDINATE ,observed_coordinate_matrix .shape )
        observed_coordinate_matrix =observed_coordinate_matrix +coordinate_noise 

    else :
    # 如果后面想让 fig4d 也加噪声，打开下面两行即可
    # noise = np.random.normal(0, noise_std, mar.shape)
    # mar = mar + noise
        pass 

    observable_matrix_H =build_observable_matrix_H (observed_coordinate_matrix ,OBSERVABLE_WINDOW_LENGTH ,DELAY_TAU )

    print ("hankel shape:",observable_matrix_H .shape )

    left_singular_vectors_U ,singular_values ,right_singular_vectors_Vh =np .linalg .svd (observable_matrix_H ,full_matrices =False )

    total_sum =np .sum (singular_values )
    cumulative_sum =np .cumsum (singular_values )
    retained_rank_r =np .argmax (cumulative_sum >=0.99 *total_sum )+1 

    print ("99% singular values number:",retained_rank_r )

    singular_values_squared_sum =np .sum (singular_values **2 )
    singular_values_proportions =(singular_values **2 )/singular_values_squared_sum 

    for i ,singular_value_value in enumerate (singular_values [:10 ]):
        proportion =singular_values_proportions [i ]
        print (f"Singular value {i +1 }: {singular_value_value }, Proportion: {proportion }")

    retained_rank_r =min (RETAINED_RANK_R ,len (singular_values ))
    reduced_coordinates_V =np .diag (singular_values [0 :retained_rank_r ])@right_singular_vectors_Vh [0 :retained_rank_r ,:]

    dt =time_grid_t [1 ]-time_grid_t [0 ]

    reduced_velocity_V_dot =np .zeros_like (reduced_coordinates_V )

    for i in range (retained_rank_r ):
        reduced_velocity_V_dot [i ,:]=np .gradient (reduced_coordinates_V [i ,:],dt ,edge_order =2 )

        # =====================================================
        # dv_dt 加噪声部分：目前不启用，保留给后面使用
        # =====================================================
        # noise_std_dv = 0.1
        # noise_dv = np.random.normal(0, noise_std_dv, dv_dt.shape)
        # dv_dt = dv_dt + noise_dv

    closure_defect_beta_values =[]
    A_t_column_norm_values =[]

    for i in range (0 ,OBSERVABLE_WINDOW_LENGTH -WINDOW_T ,1 ):
        start =i 
        end =start +WINDOW_T 

        V_dot_t =reduced_velocity_V_dot [:,start :end ]
        V_t =reduced_coordinates_V [:,start :end ]

        A_t =V_dot_t @np .linalg .pinv (V_t )

        A_t_column_norms =np .linalg .norm (A_t ,axis =0 )
        A_t_column_norm_mean =np .mean (A_t_column_norms )

        R_t =V_dot_t -A_t @V_t 

        R_t_row_norms =np .linalg .norm (R_t ,axis =1 )
        beta_t =np .mean (R_t_row_norms )

        A_t_column_norm_values .append (A_t_column_norm_mean )
        closure_defect_beta_values .append (beta_t )

    closure_defect_beta_values =np .array (closure_defect_beta_values )
    A_t_column_norm_values =np .array (A_t_column_norm_values )

    x_start =data_start +WINDOW_T //2 
    x_end =x_start +len (closure_defect_beta_values )
    closure_defect_beta_time_grid =time_grid_t [x_start :x_end ]

    if len (closure_defect_beta_time_grid )!=len (closure_defect_beta_values ):
        closure_defect_beta_time_grid =np .arange (len (closure_defect_beta_values ))*dt +time_grid_t [x_start ]

    return closure_defect_beta_time_grid ,closure_defect_beta_values ,A_t_column_norm_values ,observed_coordinate_matrix 

    # =========================================================
    # 6. 绘图格式函数
    # =========================================================
def format_beta_axis (ax ,show_xlabel =False ):
    ax .set_ylabel (r'$\beta$',fontsize =LABEL_SIZE ,color ='black')

    if show_xlabel :
        ax .set_xlabel (r'$t$',fontsize =LABEL_SIZE )
        ax .xaxis .set_label_coords (0.5 ,-0.35 )
    else :
        ax .set_xlabel ("")
        ax .tick_params (labelbottom =False )

    ax .tick_params (axis ='x',labelsize =TICK_SIZE ,width =SPINE_WIDTH ,length =3 )
    ax .tick_params (axis ='y',labelsize =TICK_SIZE ,width =SPINE_WIDTH ,length =3 )
    ax .xaxis .set_major_locator (MaxNLocator (nbins =5 ))

    ax .spines ['top'].set_visible (False )
    ax .spines ['right'].set_visible (False )

    ax .spines ['left'].set_visible (True )
    ax .spines ['bottom'].set_visible (True )

    ax .spines ['left'].set_linewidth (SPINE_WIDTH )
    ax .spines ['bottom'].set_linewidth (SPINE_WIDTH )


def save_figure (fig ,filename ):
    output_path =os .path .join (OUTPUT_DIR ,f"{filename }.svg")
    fig .savefig (output_path ,format ="svg",dpi =600 ,facecolor ="white")
    print (f"Saved: {output_path }")

    # =========================================================
    # 7. fig4d：三种区域，无噪声，不显示 legend
    #    边界与 window 图严格对齐
    # =========================================================
fig =plt .figure (figsize =(FIG_W ,FIG_H_FIG4D ),dpi =DPI_SHOW )

GAP_4D =0.105 
SINGLE_H =0.205 
BOTTOMS_4D =[0.755 ,0.445 ,PANEL_BOTTOM ]

axes =[]

for bottom_i in BOTTOMS_4D :
    ax =fig .add_axes ([BETA_LEFT ,bottom_i ,BETA_WIDTH ,SINGLE_H ])
    axes .append (ax )

for ax ,region_case in zip (axes ,OBSERVED_REGION_CASES ):
    print ("\nProcessing fig4d:",region_case ["name"])

    closure_defect_beta_time_grid ,closure_defect_beta ,_ ,observed_coordinate_matrix =compute_closure_defect_beta_from_rows (
    flattened_coordinate_matrix =flattened_coordinate_matrix ,
    time_grid_t =time_grid_t ,
    selected_coordinate_row_ranges =region_case ["rows"],
    data_start =ANALYSIS_START_INDEX ,
    data_end =ANALYSIS_END_INDEX ,
    OBSERVABLE_WINDOW_LENGTH =OBSERVABLE_WINDOW_LENGTH ,
    DELAY_TAU =DELAY_TAU ,
    RETAINED_RANK_R =RETAINED_RANK_R ,
    WINDOW_T =WINDOW_T ,
    add_noise =False ,
    NOISE_STD_COORDINATE =NOISE_STD ,
    random_seed =RANDOM_SEED 
    )

    ax .plot (
    closure_defect_beta_time_grid ,
    closure_defect_beta ,
    linestyle ='-',
    color =region_case ["color"],
    linewidth =LINE_WIDTH ,
    alpha =1.0 
    )

for i ,ax in enumerate (axes ):
    format_beta_axis (ax ,show_xlabel =(i ==len (axes )-1 ))

save_figure (fig ,"fig4d")
plt .show ()

# =========================================================
# 8. figs3：只有 CORE 区域，加噪声，不显示 legend
#    边界与 window 图严格对齐
# =========================================================
core_region_case =OBSERVED_REGION_CASES [2 ]

print ("\nProcessing figs3: CORE with noise")

closure_defect_beta_noise_time_grid ,closure_defect_beta_noise ,_ ,observed_coordinate_matrix =compute_closure_defect_beta_from_rows (
flattened_coordinate_matrix =flattened_coordinate_matrix ,
time_grid_t =time_grid_t ,
selected_coordinate_row_ranges =core_region_case ["rows"],
data_start =ANALYSIS_START_INDEX ,
data_end =ANALYSIS_END_INDEX ,
OBSERVABLE_WINDOW_LENGTH =OBSERVABLE_WINDOW_LENGTH ,
DELAY_TAU =DELAY_TAU ,
RETAINED_RANK_R =RETAINED_RANK_R ,
WINDOW_T =WINDOW_T ,
add_noise =True ,
NOISE_STD_COORDINATE =NOISE_STD ,
random_seed =RANDOM_SEED 
)

fig ,ax1 =plt .subplots (figsize =(FIG_W ,FIG_H_FIGS3 ),dpi =DPI_SHOW )

# =========================
# 左轴：β(t)
# =========================
ax1 .plot (
closure_defect_beta_noise_time_grid ,
closure_defect_beta_noise ,
color ='red',
linewidth =LINE_WIDTH ,
label =r'$\beta(t)$'
)

ax1 .set_ylabel (r'$\beta(t)$',fontsize =LABEL_SIZE ,color ='black')
ax1 .tick_params (axis ='y')

format_beta_axis (ax1 ,show_xlabel =True )
ax1 .xaxis .set_label_coords (0.5 ,-0.26 )

# =========================
# 右轴：原始 noisy signal
# =========================
ax2 =ax1 .twinx ()

ax2 .plot (
closure_defect_beta_noise_time_grid ,
observed_coordinate_matrix [0 ,:len (closure_defect_beta_noise_time_grid )],# ⭐关键：对齐长度
linestyle ='--',
color ='gray',
linewidth =0.8 ,
alpha =0.5 ,
label ='signal'
)

ax2 .set_ylabel ('signal',fontsize =LABEL_SIZE ,color ='gray')
ax2 .tick_params (axis ='y',colors ='gray',labelsize =LABEL_SIZE )

# =========================
# 保存
# =========================
save_figure (fig ,"figs3")
plt .show ()
# ====================================================================================================================
# =========================================================
# 9. fig4c：两个窗口下的 ADK 结果
#    直接追加在前面代码后面即可
# =========================================================
from matplotlib .ticker import MaxNLocator 

# ---------- fig4c 参数 ----------
FIG_H_FIG4C =6.0 

# 两个窗口
# Use the maximum windows obtained from Fig. 4(b).
WINDOW_T1 =WINDOW_T1 
WINDOW_T2 =WINDOW_T2 

# fig4c 使用全区域
selected_coordinate_row_ranges_fig4c =[(0 ,642 )]

# fig4c 使用前 2 个奇异值
RETAINED_RANK_R_FIG4C =2 

# 画布与边界：保持和前面的 fig4d / window 对齐
GAP_4C =0.095 
AX_H_4C =(PANEL_TOP -PANEL_BOTTOM -GAP_4C )/2 
BOTTOM_LOW =PANEL_BOTTOM 
BOTTOM_HIGH =PANEL_BOTTOM +AX_H_4C +GAP_4C 
XLABEL_Y_4C =0.05325 

LEGEND_SIZE =20 
BETA_COLOR ="#C42238"
NMP_COLOR ="purple"
LID_COLOR ="blue"


def compute_closure_defect_beta_two_windows_from_rows (
flattened_coordinate_matrix ,
time_grid_t ,
selected_coordinate_row_ranges ,
data_start ,
data_end ,
OBSERVABLE_WINDOW_LENGTH ,
DELAY_TAU ,
RETAINED_RANK_R ,
window_lengths_T ,
add_noise =False ,
NOISE_STD_COORDINATE =0.2 ,
random_seed =None 
):
    observed_coordinate_matrix_parts =[
    flattened_coordinate_matrix [start :end ,:]
    for start ,end in selected_coordinate_row_ranges 
    ]

    n_columns_by_selected_part =[part .shape [1 ]for part in observed_coordinate_matrix_parts ]

    if len (set (n_columns_by_selected_part ))!=1 :
        raise ValueError ("All selected parts must have the same number of columns.")

    observed_coordinate_matrix =np .vstack (observed_coordinate_matrix_parts )
    observed_coordinate_matrix =observed_coordinate_matrix [:,data_start :data_end ]

    # 噪声部分：当前默认不加，保留给后面使用
    if add_noise :
        if random_seed is not None :
            np .random .seed (random_seed )

        coordinate_noise =np .random .normal (0 ,NOISE_STD_COORDINATE ,observed_coordinate_matrix .shape )
        observed_coordinate_matrix =observed_coordinate_matrix +coordinate_noise 

    else :
    # 如果后面想给 fig4c 加噪声，打开下面两行即可
    # noise = np.random.normal(0, noise_std, mar.shape)
    # mar = mar + noise
        pass 

    observable_matrix_H =build_observable_matrix_H (observed_coordinate_matrix ,OBSERVABLE_WINDOW_LENGTH ,DELAY_TAU )
    print ("fig4c hankel shape:",observable_matrix_H .shape )

    left_singular_vectors_U ,singular_values ,right_singular_vectors_Vh =np .linalg .svd (observable_matrix_H ,full_matrices =False )

    singular_values_squared_sum =np .sum (singular_values **2 )
    singular_values_proportions =(singular_values **2 )/singular_values_squared_sum 

    for i ,singular_value_value in enumerate (singular_values [:10 ]):
        proportion =singular_values_proportions [i ]
        print (f"fig4c Singular value {i +1 }: {singular_value_value }, Proportion: {proportion }")

    retained_rank_r =min (RETAINED_RANK_R ,len (singular_values ))
    reduced_coordinates_V =np .diag (singular_values [0 :retained_rank_r ])@right_singular_vectors_Vh [0 :retained_rank_r ,:]

    dt =time_grid_t [1 ]-time_grid_t [0 ]

    reduced_velocity_V_dot =np .zeros_like (reduced_coordinates_V )

    for i in range (retained_rank_r ):
        reduced_velocity_V_dot [i ,:]=np .gradient (reduced_coordinates_V [i ,:],dt ,edge_order =2 )

        # dv_dt 加噪声部分：目前不启用，保留给后面使用
        # noise_std_dv = 0.1
        # noise_dv = np.random.normal(0, noise_std_dv, dv_dt.shape)
        # dv_dt = dv_dt + noise_dv

    closure_defect_beta_by_T ={}

    for window_length_T_now in window_lengths_T :
        closure_defect_beta_values =[]
        A_t_column_norm_values =[]

        for i in range (0 ,OBSERVABLE_WINDOW_LENGTH -window_length_T_now ,1 ):
            start =i 
            end =start +window_length_T_now 

            V_dot_t =reduced_velocity_V_dot [:,start :end ]
            V_t =reduced_coordinates_V [:,start :end ]

            A_t =V_dot_t @np .linalg .pinv (V_t )

            A_t_column_norms =np .linalg .norm (A_t ,axis =0 )
            A_t_column_norm_mean =np .mean (A_t_column_norms )

            R_t =V_dot_t -A_t @V_t 

            R_t_row_norms =np .linalg .norm (R_t ,axis =1 )
            beta_t =np .mean (R_t_row_norms )

            A_t_column_norm_values .append (A_t_column_norm_mean )
            closure_defect_beta_values .append (beta_t )

        closure_defect_beta_by_T [window_length_T_now ]={
        "col":np .array (A_t_column_norm_values ),
        "row":np .array (closure_defect_beta_values )
        }

    return closure_defect_beta_by_T 


def set_closure_defect_beta_ylim_for_legend (ax ,closure_defect_beta ,top_pad =0.35 ,bottom_pad =0.08 ):
    closure_defect_beta =np .asarray (closure_defect_beta )
    closure_defect_beta =closure_defect_beta [np .isfinite (closure_defect_beta )]

    if closure_defect_beta .size ==0 :
        return 

    ymin =np .nanmin (closure_defect_beta )
    ymax =np .nanmax (closure_defect_beta )
    yrange =ymax -ymin 

    if yrange ==0 :
        yrange =max (abs (ymax ),1.0 )

    ax .set_ylim (
    ymin -bottom_pad *yrange ,
    ymax +top_pad *yrange 
    )


def format_beta_left_axis_4c (ax ,show_xlabel_tick =False ):
    ax .set_ylabel (r'$\beta$',fontsize =LABEL_SIZE ,labelpad =8 )
    ax .tick_params (axis ="both",labelsize =TICK_SIZE ,labelcolor ="black")
    ax .tick_params (direction ="out",length =3.5 ,width =SPINE_WIDTH ,pad =3 )

    if not show_xlabel_tick :
        ax .tick_params (labelbottom =False )

    ax .xaxis .set_major_locator (MaxNLocator (nbins =5 ))
    ax .yaxis .set_major_locator (MaxNLocator (nbins =3 ))

    ax .spines ["top"].set_visible (False )
    ax .spines ["right"].set_visible (False )
    ax .spines ["left"].set_linewidth (SPINE_WIDTH )
    ax .spines ["bottom"].set_linewidth (SPINE_WIDTH )

    ax .grid (False )


def format_angle_right_axis_4c (ax ):
    ax .set_ylabel ("Angle",fontsize =LABEL_SIZE ,labelpad =8 )
    ax .tick_params (axis ="y",labelsize =TICK_SIZE ,labelcolor ="black")
    ax .tick_params (direction ="out",length =3.5 ,width =SPINE_WIDTH ,pad =3 )

    ax .yaxis .set_major_locator (MaxNLocator (nbins =3 ))

    ax .spines ["top"].set_visible (False )
    ax .spines ["left"].set_visible (False )
    ax .spines ["right"].set_linewidth (SPINE_WIDTH )
    ax .spines ["bottom"].set_visible (False )

    ax .set_ylim (0 ,5 )
    ax .grid (False )


def add_combined_legend_4c (ax_left ,ax_right ):
    lines_l ,labels_l =ax_left .get_legend_handles_labels ()
    lines_r ,labels_r =ax_right .get_legend_handles_labels ()

    leg =ax_left .legend (
    lines_l +lines_r ,
    labels_l +labels_r ,
    loc ="upper right",
    fontsize =LEGEND_SIZE ,
    frameon =True ,
    framealpha =0.90 ,
    facecolor ="white",
    edgecolor ="none",
    handlelength =1.6 ,
    borderpad =0.25 ,
    labelspacing =0.25 
    )

    leg .set_zorder (30 )


def save_fig4c (fig ):
    output_path =os .path .join (OUTPUT_DIR ,"fig4c.svg")
    fig .savefig (output_path ,format ="svg",dpi =DPI_SHOW ,facecolor ="white",transparent =False )
    print (f"Saved: {output_path }")


    # =========================================================
    # 计算 fig4c 的两个窗口 beta
    # =========================================================
closure_defect_beta_4c_by_T =compute_closure_defect_beta_two_windows_from_rows (
flattened_coordinate_matrix =flattened_coordinate_matrix ,
time_grid_t =time_grid_t ,
selected_coordinate_row_ranges =selected_coordinate_row_ranges_fig4c ,
data_start =ANALYSIS_START_INDEX ,
data_end =ANALYSIS_END_INDEX ,
OBSERVABLE_WINDOW_LENGTH =OBSERVABLE_WINDOW_LENGTH ,
DELAY_TAU =DELAY_TAU ,
RETAINED_RANK_R =RETAINED_RANK_R_FIG4C ,
window_lengths_T =[WINDOW_T1 ,WINDOW_T2 ],
add_noise =False ,
NOISE_STD_COORDINATE =NOISE_STD ,
random_seed =RANDOM_SEED 
)

closure_defect_beta1_values =closure_defect_beta_4c_by_T [WINDOW_T1 ]["row"]
closure_defect_beta2_values =closure_defect_beta_4c_by_T [WINDOW_T2 ]["row"]

# =========================================================
# 绘制 fig4c
# =========================================================
fig =plt .figure (figsize =(FIG_W ,FIG_H_FIG4C ),dpi =DPI_SHOW ,facecolor ="white")
fig .patch .set_facecolor ("white")

angle_time_grid_t =time_grid_t [0 :ANALYSIS_END_INDEX -ANALYSIS_START_INDEX ]

# ---------- 上图：step_a ----------
ax1_left =fig .add_axes ([BETA_LEFT ,BOTTOM_HIGH ,BETA_WIDTH ,AX_H_4C ],facecolor ="white")
ax1_right =ax1_left .twinx ()

closure_defect_beta1_time_grid =time_grid_t [0 :len (closure_defect_beta1_values )]

ax1_left .plot (
closure_defect_beta1_time_grid ,
closure_defect_beta1_values ,
linestyle ="-",
label =r'$\beta_1$ ($T_1$)',
color =BETA_COLOR ,
alpha =1.0 ,
linewidth =LINE_WIDTH ,
zorder =10 
)

ax1_right .plot (
angle_time_grid_t ,
angle_theta_NMP [ANALYSIS_START_INDEX :ANALYSIS_END_INDEX ],
color =NMP_COLOR ,
alpha =0.70 ,
linewidth =LINE_WIDTH 
)

ax1_right .plot (
angle_time_grid_t ,
angle_theta_LID [ANALYSIS_START_INDEX :ANALYSIS_END_INDEX ],
color =LID_COLOR ,
alpha =0.70 ,
linewidth =LINE_WIDTH 
)

set_closure_defect_beta_ylim_for_legend (ax1_left ,closure_defect_beta1_values ,top_pad =0.35 ,bottom_pad =0.08 )

ax1_left .set_xlim (0 ,22000 )
ax1_right .set_xlim (0 ,22000 )

format_beta_left_axis_4c (ax1_left ,show_xlabel_tick =False )
format_angle_right_axis_4c (ax1_right )
add_combined_legend_4c (ax1_left ,ax1_right )

# ---------- 下图：step_b ----------
ax2_left =fig .add_axes ([BETA_LEFT ,BOTTOM_LOW ,BETA_WIDTH ,AX_H_4C ],facecolor ="white")
ax2_right =ax2_left .twinx ()

closure_defect_beta2_time_grid =time_grid_t [0 :len (closure_defect_beta2_values )]

ax2_left .plot (
closure_defect_beta2_time_grid ,
closure_defect_beta2_values ,
linestyle ="-",
label =r'$\beta_2$ ($T_2$)',
color =BETA_COLOR ,
alpha =1.0 ,
linewidth =LINE_WIDTH ,
zorder =10 
)

ax2_right .plot (
angle_time_grid_t ,
angle_theta_NMP [ANALYSIS_START_INDEX :ANALYSIS_END_INDEX ],
color =NMP_COLOR ,
alpha =0.70 ,
linewidth =LINE_WIDTH 
)

ax2_right .plot (
angle_time_grid_t ,
angle_theta_LID [ANALYSIS_START_INDEX :ANALYSIS_END_INDEX ],
color =LID_COLOR ,
alpha =0.70 ,
linewidth =LINE_WIDTH 
)

set_closure_defect_beta_ylim_for_legend (ax2_left ,closure_defect_beta2_values ,top_pad =0.35 ,bottom_pad =0.08 )

ax2_left .set_xlim (0 ,22000 )
ax2_right .set_xlim (0 ,22000 )

format_beta_left_axis_4c (ax2_left ,show_xlabel_tick =True )
format_angle_right_axis_4c (ax2_right )
add_combined_legend_4c (ax2_left ,ax2_right )

# 统一 x 轴标签
ax2_left .set_xlabel (r"$t$",fontsize =LABEL_SIZE ,labelpad =5 )
ax2_left .xaxis .set_label_coords (0.5 ,-0.19 )

save_fig4c (fig )
plt .show (block =True )

print ("Finished. SVG figures are saved in:",OUTPUT_DIR )
