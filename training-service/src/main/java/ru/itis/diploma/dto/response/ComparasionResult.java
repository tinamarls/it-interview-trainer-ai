package ru.itis.diploma.dto.response;

import lombok.Data;

@Data
public class ComparasionResult {
    private String error;
    private String fullResultJson;
    private Double matchingPercent;
    private String neededSkills;
}