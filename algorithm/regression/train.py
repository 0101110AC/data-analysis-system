from .models import (
    LinearRegressionModel,
    RidgeRegressionModel,
    LassoRegressionModel,
    ElasticNetRegressionModel,
    RegressionParams
)
from ...data.regression_data import RegressionDataManager

def train_and_evaluate(dataset_name='california_housing'):
    # 根据数据集名称加载数据
    if dataset_name == 'california_housing':
        X_train, X_test, y_train, y_test = RegressionDataManager.load_california_housing()
    elif dataset_name == 'linear':
        X_train, X_test, y_train, y_test = RegressionDataManager.load_linear_regression()
    elif dataset_name == 'nonlinear':
        X_train, X_test, y_train, y_test = RegressionDataManager.load_nonlinear_regression()
    else:
        raise ValueError(f"未知的数据集名称: {dataset_name}")

    
    # 定义要测试的算法和参数
    algorithms = [
        (LinearRegressionModel, RegressionParams(fit_intercept=True, copy_X=True)),
        (RidgeRegressionModel, RegressionParams(alpha=1.0)),
        (LassoRegressionModel, RegressionParams(alpha=1.0)),
        (ElasticNetRegressionModel, RegressionParams(alpha=1.0, l1_ratio=0.5))
    ]
    
    results = {}
    
    for Algorithm, params in algorithms:
        algo_name = Algorithm.__name__
        print(f"\n训练 {algo_name}...")
        
        # 初始化模型
        model = Algorithm(params)
        
        try:
            # 训练模型
            history = model.train(X_train, y_train)
            
            # 评估模型
            train_metrics = model.evaluate(X_train, y_train)
            test_metrics = model.evaluate(X_test, y_test)
            
            # 保存结果
            results[algo_name] = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'history': history,
                'params': model.get_params()
            }
            
            # 打印评估结果
            print(f"训练集评估结果:")
            print(f"MSE: {train_metrics['mse']:.4f}")
            print(f"RMSE: {train_metrics['rmse']:.4f}")
            print(f"R2: {train_metrics['r2']:.4f}")
            
            print(f"\n测试集评估结果:")
            print(f"MSE: {test_metrics['mse']:.4f}")
            print(f"RMSE: {test_metrics['rmse']:.4f}")
            print(f"R2: {test_metrics['r2']:.4f}")
            
        finally:
            model.dispose()
    
    return results

def main():
    try:
        # 对每个数据集进行训练和评估
        datasets = ['california_housing', 'linear', 'nonlinear']
        
        for dataset in datasets:
            print(f"\n使用数据集: {dataset}")
            results = train_and_evaluate(dataset)
        
        # 比较不同模型的性能
        print("\n模型性能比较：")
        for algo_name, result in results.items():
            print(f"\n{algo_name}:")
            print(f"测试集 R2 分数: {result['test_metrics']['r2']:.4f}")
            print(f"测试集 RMSE: {result['test_metrics']['rmse']:.4f}")
    
    except Exception as e:
        print(f"训练过程中出现错误：{str(e)}")

if __name__ == "__main__":
    main()