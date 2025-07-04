package ru.itis.diploma.models.hr;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import org.hibernate.annotations.JdbcTypeCode;
import ru.itis.diploma.models.User;

import java.sql.Types;
import java.time.LocalDateTime;
import java.util.List;

@Data
@RequiredArgsConstructor
@AllArgsConstructor
@Builder
@Entity
@Table(name = "interview_attempts")
public class HrAttempt {
    @Id
    @Column(name = "id")
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @Column(name = "start_time", nullable = false)
    private LocalDateTime startTime;

    @Column(name = "end_time")
    private LocalDateTime endTime;

    @Column(name = "total_score")
    private Double totalScore;

    @Column(name = "status", nullable = false)
    @Enumerated(EnumType.STRING)
    private AttemptStatus status;

    @Column(name = "feedback", columnDefinition = "TEXT")
    private String feedback;

    @JdbcTypeCode(Types.VARBINARY)
    @Column(name = "video_image_data", columnDefinition = "BYTEA")
    private byte[] videoImageData;

    @JdbcTypeCode(Types.VARBINARY)
    @Column(name = "audio_image_data", columnDefinition = "BYTEA")
    private byte[] audioImageData;

    public enum AttemptStatus {
        IN_PROGRESS,
        COMPLETED,
        ABORTED
    }
}
