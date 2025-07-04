package ru.itis.diploma.dto.request;

import java.time.LocalDateTime;
import java.util.UUID;

public record ResumeDto(UUID id, String fileName, LocalDateTime uploadedAt) {}

