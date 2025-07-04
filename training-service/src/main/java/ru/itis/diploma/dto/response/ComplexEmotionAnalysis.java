package ru.itis.diploma.dto.response;

import lombok.Data;
import java.util.Map;

@Data
public class ComplexEmotionAnalysis {
    private Map<String, Object> comprehensiveAssessment;
    private String emotionTimelineGraphFile;
    private String error; // для ошибки этого анализа
}