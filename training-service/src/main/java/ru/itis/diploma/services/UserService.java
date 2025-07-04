package ru.itis.diploma.services;

import lombok.RequiredArgsConstructor;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.stereotype.Service;
import ru.itis.diploma.models.User;
import ru.itis.diploma.repositories.UserRepository;

import java.util.Optional;

@Service
@RequiredArgsConstructor
public class UserService implements UserDetailsService {

    private final UserRepository userRepository;

    @Override
    public UserDetails loadUserByUsername(String username) {
        User userFromDB = userRepository.findByEmail(username);
        if (userFromDB == null) {
            throw  new IllegalArgumentException("User not found");
        }
        return userFromDB;
    }

    public User getUserByEmail(String email) {
        User userFromDB = userRepository.findByEmail(email);
        if (userFromDB == null) {
            throw  new IllegalArgumentException("User not found");
        }
        return userFromDB;
    }

    public User findUserById(Long userId) {
        Optional<User> userFromDb = userRepository.findById(userId);
        return userFromDb.orElse(new User());
    }

    public Long saveUser(User user) {
        User userFromDB = userRepository.findByEmail(user.getEmail());

        if (userFromDB != null) {
            throw new IllegalArgumentException("Email already registered");
        }

        return userRepository.save(user).getId();
    }

    public boolean deleteUser(Long userId) {
        if (userRepository.findById(userId).isPresent()) {
            userRepository.deleteById(userId);
            return true;
        }
        return false;
    }

    /**
     * Получение пользователя по имени пользователя
     * <p>
     * Нужен для Spring Security
     *
     * @return пользователь
     */
    public UserDetailsService userDetailsService() {
        return this::loadUserByUsername;
    }

}
