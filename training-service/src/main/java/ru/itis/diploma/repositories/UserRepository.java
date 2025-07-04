package ru.itis.diploma.repositories;

import org.springframework.data.jpa.repository.JpaRepository;
import ru.itis.diploma.models.User;

public interface UserRepository extends JpaRepository<User, Long> {
    User findByEmail(String email);
}
