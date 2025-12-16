import pandas as pd
import numpy as np


# =========================
# 列名映射（英文 -> 中文）
# =========================
COLUMN_MAPPING = {
    "CustomerID": "客户唯一标识",
    "Churn": "是否流失",

    "MonthlyRevenue": "月均消费金额",
    "MonthlyMinutes": "月通话分钟数",
    "TotalRecurringCharge": "每月固定套餐费用",
    "DirectorAssistedCalls": "人工协助拨号次数",
    "OverageMinutes": "超出套餐的通话分钟数",
    "RoamingCalls": "漫游通话次数",
    "PercChangeMinutes": "通话分钟数环比变化率",
    "PercChangeRevenues": "消费金额环比变化率",

    "DroppedCalls": "掉线次数",
    "BlockedCalls": "被阻塞的通话次数",
    "UnansweredCalls": "未接通电话次数",
    "CustomerCareCalls": "客服热线拨打次数",
    "ThreewayCalls": "三方通话次数",
    "ReceivedCalls": "接听电话次数",
    "OutboundCalls": "呼出电话次数",
    "InboundCalls": "呼入电话次数",

    "PeakCallsInOut": "高峰时段通话次数",
    "OffPeakCallsInOut": "非高峰时段通话次数",
    "DroppedBlockedCalls": "掉线与阻塞通话总次数",

    "CallForwardingCalls": "呼叫转移使用次数",
    "CallWaitingCalls": "呼叫等待使用次数",

    "MonthsInService": "在网时长（月）",
    "UniqueSubs": "唯一订阅数量",
    "ActiveSubs": "当前激活订阅数量",

    "ServiceArea": "服务区域",

    "Handsets": "拥有手机数量",
    "HandsetModels": "使用过的手机型号数",
    "CurrentEquipmentDays": "当前设备使用天数",

    "AgeHH1": "家庭成员1年龄",
    "AgeHH2": "家庭成员2年龄",
    "ChildrenInHH": "家庭是否有儿童",

    "HandsetRefurbished": "是否使用翻新机",
    "HandsetWebCapable": "手机是否支持上网",

    "TruckOwner": "是否拥有卡车",
    "RVOwner": "是否拥有房车",
    "Homeownership": "是否拥有住房",

    "BuysViaMailOrder": "是否通过邮购方式消费",
    "RespondsToMailOffers": "是否响应过邮寄营销",
    "OptOutMailings": "是否拒绝接收邮件营销",

    "NonUSTravel": "是否有非美国出行经历",
    "OwnsComputer": "是否拥有电脑",
    "HasCreditCard": "是否持有信用卡",

    "RetentionCalls": "挽留部门通话次数",
    "RetentionOffersAccepted": "是否接受挽留优惠",
    "MadeCallToRetentionTeam": "是否联系过挽留团队",

    "NewCellphoneUser": "是否新手机用户",
    "NotNewCellphoneUser": "是否非新手机用户",

    "ReferralsMadeBySubscriber": "推荐他人入网次数",

    "IncomeGroup": "收入水平分组",
    "OwnsMotorcycle": "是否拥有摩托车",
    "AdjustmentsToCreditRating": "信用评级调整次数",
    "HandsetPrice": "手机价格",

    "CreditRating": "信用评级",
    "PrizmCode": "PRIZM市场细分编码",
    "Occupation": "职业类别",
    "MaritalStatus": "婚姻状况",
}


# =========================
# 预处理主函数
# =========================
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    对 Cell2Cell 数据集进行统一预处理
    返回可直接用于建模的 DataFrame
    """

    df = df.copy()

    # 1. 重命名列
    df.rename(columns=COLUMN_MAPPING, inplace=True)

    # 2. Yes / No → 1 / 0
    yes_no_cols = [
        "是否流失",
        "家庭是否有儿童",
        "是否使用翻新机",
        "手机是否支持上网",
        "是否拥有卡车",
        "是否拥有房车",
        "是否通过邮购方式消费",
        "是否响应过邮寄营销",
        "是否拒绝接收邮件营销",
        "是否有非美国出行经历",
        "是否拥有电脑",
        "是否持有信用卡",
        "是否新手机用户",
        "是否非新手机用户",
        "是否拥有摩托车",
        "是否联系过挽留团队",
        "是否接受挽留优惠",
    ]

    for col in yes_no_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"yes": 1, "no": 0})
                .astype("Int64")
            )
    # 是否接受挽留优惠（结构性缺失）
    df["是否接受挽留优惠"] = (
        df["是否接受挽留优惠"]
        .fillna(0)
        .astype(int)
    )

    # 3. Known / Unknown（二元）
    if "是否拥有住房" in df.columns:
        df["是否拥有住房"] = (
            df["是否拥有住房"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"known": 1, "unknown": 0})
            .fillna(0)
            .astype("Int64")
        )

    # 4. 信用评级：如 "1-Highest" → 1
    if "信用评级" in df.columns:
        df["信用评级"] = (
            df["信用评级"]
            .astype(str)
            .str.extract(r"(\d+)")
            .astype(float)
        )

    # 5. 手机价格（Unknown → NaN → median）
    if "手机价格" in df.columns:
        df["手机价格"] = pd.to_numeric(
            df["手机价格"], errors="coerce"
        )
        df["手机价格"] = df["手机价格"].fillna(df["手机价格"].median())

    # 6. 数值列缺失值统一用 median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # 7. One-Hot 编码（高基数类别）
    cat_cols = [
        col for col in ["PRIZM市场细分编码", "职业类别", "婚姻状况"]
        if col in df.columns
    ]

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df