package ru.itis.diploma.models;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.RequiredArgsConstructor;

import java.time.LocalDateTime;
import java.util.UUID;

@Data
@RequiredArgsConstructor
@AllArgsConstructor
@Builder
@Entity
@Table(name = "resume")
public class Resume {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    private UUID id;

    @ManyToOne
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @Column
    private String fileName;

    @Column
    private String filePath;

    @Column(columnDefinition = "TEXT")
    private String extractedText;

    private LocalDateTime uploadedAt;
}

