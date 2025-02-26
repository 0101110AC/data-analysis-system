from app.data.datasets import DatasetManager
import os

def main():
    # 确保data目录存在
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # 初始化数据管理器
    dm = DatasetManager('data')
    
    print('开始生成数据集...')
    
    # 生成各类数据集
    print('生成聚类数据集...')
    dm.generate_clustering_datasets()
    
    print('生成分类数据集...')
    dm.generate_classification_datasets()
    
    print('生成回归数据集...')
    dm.generate_regression_datasets()
    
    print('数据集生成完成！')

if __name__ == '__main__':
    main()