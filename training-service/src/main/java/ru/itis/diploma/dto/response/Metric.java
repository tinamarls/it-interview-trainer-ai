package ru.itis.diploma.dto.response;

import lombok.Data;

@Data
public class Metric {
    private String name;
    private String key;
    private double value;
    private double score;
    private String interpretation;
    private String recommendation;
}