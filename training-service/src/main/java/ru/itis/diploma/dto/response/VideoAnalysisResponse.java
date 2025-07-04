package ru.itis.diploma.dto.response;

import lombok.Data;

@Data
public class VideoAnalysisResponse {
    private String result;
    private String videoImagePath;
    private String audioImagePath;
    private String error; // для общей ошибки
}