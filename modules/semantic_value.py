import torch
from torch import nn

from config import Config
from data.semantic_cache import COLUMN_LAT, COLUMN_LON, clean_value


# 保留经纬度的合法范围。
LAT_LOW, LAT_HIGH = -90.0, 90.0
LON_LOW, LON_HIGH = -180.0, 180.0


# 保留原有经纬度编码及投影，单独提供数值语义表示。
class SemanticValue(nn.Module):
    """
    功能描述:
        使用原有多频率正余弦编码和线性层处理用户、服务经纬度。
    参数:
        cfg：Config，实体数量、字段维度和经纬度频率配置，标量。
    返回值:
        SemanticValue：前向返回用户、服务数值表示及两侧坐标有效性。
    """

    # 初始化保持原样的经纬度投影及运行时缓存。
    def __init__(self, cfg: Config) -> None:
        """
        功能描述:
            校验经纬度配置并创建原有线性投影，不包含 ID 查表。
        参数:
            cfg：Config，数值语义配置，标量。
        返回值:
            None：完成投影和缓存初始化。
        """
        super().__init__()
        self.field_dim = int(cfg.semantic_dim)
        self.geo_freqs = int(cfg.semantic_geo_freqs)
        if self.field_dim < 1 or self.geo_freqs < 1:
            raise ValueError("semantic_dim 和 semantic_geo_freqs 必须为正整数。")
        self.geo_input_dim = 4 * self.geo_freqs + 1
        self.geo_projection = nn.Linear(self.geo_input_dim, self.field_dim)
        self.register_buffer("user_geo", torch.zeros(cfg.user_num, self.geo_input_dim), persistent=False)
        self.register_buffer("service_geo", torch.zeros(cfg.service_num, self.geo_input_dim), persistent=False)

    # 从原始元数据恢复经纬度缓存。
    def build_metadata(self, user_text: list[list[str]], service_text: list[list[str]]) -> None:
        """
        功能描述:
            沿用原有字段解析、正余弦编码和缺失值处理。
        参数:
            user_text：list[list[str]]，用户元数据，形状为 [U, 9]。
            service_text：list[list[str]]，服务元数据，形状为 [S, 9]。
        返回值:
            None：更新两侧经纬度特征缓存。
        """
        self.user_geo.copy_(self.geo_features(user_text))
        self.service_geo.copy_(self.geo_features(service_text))

    # 沿用原有的多频率经纬度正余弦计算。
    def geo_features(self, text: list[list[str]]) -> torch.Tensor:
        """
        功能描述:
            经纬度归一化后使用二的幂频率编码，缺失或越界时输出全零特征。
        参数:
            text：list[list[str]]，一侧实体的九列元数据，形状为 [N, 9]。
        返回值:
            torch.Tensor：单精度正余弦特征及有效性标志，形状为 [N, 4R + 1]。
        """
        # 初始化归一化坐标和经纬度有效性标志。
        lat = torch.zeros(len(text), dtype=torch.float32)
        lon = torch.zeros(len(text), dtype=torch.float32)
        has_geo = torch.zeros(len(text), dtype=torch.float32)

        # 保留原有解析、闭区间合法性判断和归一化逻辑。
        for row_idx, row in enumerate(text):
            lat_value = self.get_column_value_to_float(row, COLUMN_LAT)
            lon_value = self.get_column_value_to_float(row, COLUMN_LON)
            if lat_value is None or lon_value is None:
                continue
            if not (LAT_LOW <= lat_value <= LAT_HIGH) or not (LON_LOW <= lon_value <= LON_HIGH):
                continue
            lat[row_idx] = lat_value / 90.0
            lon[row_idx] = lon_value / 180.0
            has_geo[row_idx] = 1.0

        # 保留原有频率、正余弦顺序及缺失行清零方式。
        freqs = torch.pow(2.0, torch.arange(self.geo_freqs, dtype=torch.float32))
        lat_angle = lat.unsqueeze(-1) * freqs
        lon_angle = lon.unsqueeze(-1) * freqs
        encoding = torch.cat([
            torch.sin(lat_angle), torch.cos(lat_angle),
            torch.sin(lon_angle), torch.cos(lon_angle),
        ], dim=-1)
        encoding = encoding * has_geo.unsqueeze(-1)
        return torch.cat([encoding, has_geo.unsqueeze(-1)], dim=-1)

    # 保留经纬度路径使用的字段清洗接口。
    @staticmethod
    def get_column_value(row: list[str], column: int) -> str | None:
        """
        功能描述:
            去除首尾空白，将空串及字符串 None 识别为缺失。
        参数:
            row：list[str]，九列元数据，形状为 [9]。
            column：int，字段列号，标量。
        返回值:
            str | None：清洗后的字段文本，标量。
        """
        return clean_value(row, column)

    # 保留经纬度路径使用的浮点转换接口。
    @staticmethod
    def get_column_value_to_float(row: list[str], column: int) -> float | None:
        """
        功能描述:
            将字段转换为浮点数，缺失或无法转换时返回 None。
        参数:
            row：list[str]，九列元数据，形状为 [9]。
            column：int，字段列号，标量。
        返回值:
            float | None：浮点字段值，标量。
        """
        value = SemanticValue.get_column_value(row, column)
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    # 按实体索引执行原有经纬度线性投影，并同时返回坐标有效性掩码。
    def forward(
        self, user_indices: torch.Tensor, service_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        功能描述:
            读取经纬度缓存，返回两侧数值表示及各自的坐标有效性，缺失坐标的表示已清零。
        参数:
            user_indices：torch.Tensor，用户索引，形状为 [B]。
            service_indices：torch.Tensor，服务索引，形状为 [B]。
        返回值:
            user：用户经纬度投影表示 [B, semantic_dim]。
            service：服务经纬度投影表示 [B, semantic_dim]。
            user_valid：用户坐标是否有效 [B]。
            service_valid：服务坐标是否有效 [B]。
        """
        # 缺失坐标不携带投影偏置，避免形成伪语义证据。
        user_valid = self.user_geo[user_indices, -1] > 0
        service_valid = self.service_geo[service_indices, -1] > 0
        user = self.geo_projection(self.user_geo[user_indices]).masked_fill(~user_valid.unsqueeze(-1), 0.0)
        service = self.geo_projection(self.service_geo[service_indices]).masked_fill(~service_valid.unsqueeze(-1), 0.0)
        return user, service, user_valid, service_valid
