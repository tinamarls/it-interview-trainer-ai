package ru.itis.diploma.models.hr;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.RequiredArgsConstructor;

@Data
@RequiredArgsConstructor
@AllArgsConstructor
@Builder
@Entity
@Table(name = "hr_questions")
public class ScreeningQuestion {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String questionText;

    private String category;
//
//    @Column(name = "difficulty_level", nullable = false)
//    private Integer difficultyLevel;
//
//    @Column(name = "expected_answer_keywords", length = 1000)
//    private String expectedAnswerKeywords;
//
//    @Column(name = "time_limit_seconds")
//    private Integer timeLimitSeconds;
//
//    @Column(name = "is_active", nullable = false)
//    private Boolean isActive = true;
}
