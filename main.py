from src.config_parser import load_config
from src.utils import set_seed, setup_logger

def main():
    # 1. 加载配置
    config = load_config("configs/baseline.yaml")
    exp_name = config["experiment_name"]
    
    # 2. 初始化日志
    logger = setup_logger(exp_name)
    logger.info("============== 实验开始 ==============")
    logger.info(f"成功加载配置文件，当前算法: {config['model']['algorithm']}")
    
    # 3. 锁定随机种子
    seed = config["random_seeds"][0]
    set_seed(seed)
    logger.info(f"已全局锁定随机种子为: {seed}")
    
    logger.info("环境基建测试通过，准备进入模块二：因果发现！")
    logger.info("============== 实验结束 ==============")

if __name__ == "__main__":
    main()