package ru.itis.diploma.dto.response;

import java.time.LocalDateTime;
import java.util.UUID;

public record VacancyDto(Long id, String title, String url, LocalDateTime createdAt, String text) {}

