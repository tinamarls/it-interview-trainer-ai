package ru.itis.diploma.dto.response;

import lombok.Data;
import java.util.List;

@Data
public class FacialExpressionAnalysis {
    private List<Metric> metrics;
    private String error; // для ошибки этого анализа
}