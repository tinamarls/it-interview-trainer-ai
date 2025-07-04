package ru.itis.diploma.models.hr;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.RequiredArgsConstructor;

import java.time.LocalDateTime;

@Data
@RequiredArgsConstructor
@AllArgsConstructor
@Builder
@Entity
@Table(name = "question_responses")
public class HrResponse {
    @Id
    @Column(name = "id")
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "attempt_id", nullable = false)
    private HrAttempt interviewAttempt;

    @Column(name = "feedback", length = 1000)
    private String feedback;

    @Column(name = "response_time")
    private LocalDateTime responseTime;

//    @Column(name = "response_text", length = 4000)
//    private String responseText;
//
//    @Column(name = "video_url")
//    private String videoUrl;
//
//    @Column(name = "audio_url")
//    private String audioUrl;

//    @Column(name = "duration_seconds")
//    private Integer durationSeconds;
//
//    @Column(name = "score")
//    private Double score;
//
//    @Column(name = "confidence_score")
//    private Double confidenceScore;
//
//    @Column(name = "speech_rate")
//    private Double speechRate;
//
//    @Column(name = "hesitation_coefficient")
//    private Double hesitationCoefficient;
//
//    @Column(name = "eye_contact_score")
//    private Double eyeContactScore;
//
//    @Column(name = "facial_expression_score")
//    private Double facialExpressionScore;
//
//    @Column(name = "dominant_emotion")
//    private String dominantEmotion;
//
//    @Column(name = "keyword_match_score")
//    private Double keywordMatchScore;
}
