package ru.itis.diploma.repositories;

import org.springframework.data.jpa.repository.JpaRepository;
import ru.itis.diploma.models.Resume;
import ru.itis.diploma.models.User;

import java.util.UUID;
import java.util.List;

public interface ResumeRepository extends JpaRepository<Resume, UUID> {

    List<Resume> findAllByUserOrderByUploadedAtDesc(User user);
}
