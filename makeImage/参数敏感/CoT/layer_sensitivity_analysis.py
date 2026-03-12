"""
CMPNN层数敏感性实验可视化脚本(手动数据版)
"""
import os
import matplotlib.pyplot as plt

#尝试加载中文字体，如果服务器缺失则默认回退到无衬线字体
plt.rcParams['font.sans-serif']=['SimHei','WenQuanYi Micro Hei','WenQuanYi Zen Hei','Microsoft YaHei','sans-serif']
plt.rcParams['axes.unicode_minus']=False
plt.style.use('default')

#手动设定的真实实验数据
MANUAL_DATA={
    'Ames':[0.688,0.788,0.838,0.758,0.688],
    'BBB':[0.720,0.810,0.850,0.740,0.730],
    'CYP2D6':[0.500,0.600,0.650,0.570,0.500]
}

#颜色映射
COLORS={
    'Ames':'#d62728',
    'BBB':'#1f77b4',
    'CYP2D6':'#ff7f0e'
}

#新增形状映射(圆圈、方块、三角)
MARKERS={
    'Ames':'o',
    'BBB':'s',
    'CYP2D6':'^'
}

def plot_sensitivity_analysis(layers:list,manual_data:dict,output_dir:str):
    """绘制并保存分类任务敏感性分析图"""
    fig,ax=plt.subplots(figsize=(10,6))
    idx_layer_3=layers.index(3)
    
    for dataset,vals in manual_data.items():
        #获取当前数据集对应的形状
        m_style=MARKERS[dataset]
        
        #绘制折线图(通过marker参数动态设置形状)
        ax.plot(layers,vals,linestyle='-',marker=m_style,color=COLORS[dataset],label=f'{dataset}(AUROC)',linewidth=2,markersize=8)
        
        #标出最优的第3层(形状保持一致)
        ax.scatter(3,vals[idx_layer_3],color=COLORS[dataset],s=200,zorder=5,edgecolor='black',linewidth=2,marker=m_style)

    #设置图表属性
    ax.set_xlabel('Number of CMPNN Layers',fontsize=12)
    ax.set_ylabel('AUROC',fontsize=12)
    ax.set_xlim(0.5,5.5)
    ax.set_ylim(0.4,1.0)
    ax.set_xticks(layers)
    ax.grid(True,alpha=0.3)
    
    #将图注放在图表内部，并添加黑色直角方框
    ax.legend(loc='best',ncol=1,fontsize=10,frameon=True,edgecolor='black',fancybox=False,framealpha=1.0)

    plt.tight_layout()
    
    #确保目录存在并输出图片
    os.makedirs(output_dir,exist_ok=True)
    img_path=os.path.join(output_dir,'layer_sensitivity_classification.png')
    plt.savefig(img_path,dpi=300,bbox_inches='tight')
    plt.close()
    print(f"图片已成功保存至:{img_path}")

def main():
    #严格固定输出路径
    output_dir='/home/fsy23/UniPoly/makeImage/参数敏感/CoT'
    layers=[1,2,3,4,5]
    
    plot_sensitivity_analysis(layers,MANUAL_DATA,output_dir)

if __name__=="__main__":
    main()