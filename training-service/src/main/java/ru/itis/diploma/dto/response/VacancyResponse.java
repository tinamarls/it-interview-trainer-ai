package ru.itis.diploma.dto.response;

import com.fasterxml.jackson.annotation.JsonProperty;

public class VacancyResponse {
    @JsonProperty("extracted_text")
    public String extracted_text;
}

