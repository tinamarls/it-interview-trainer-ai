package ru.itis.diploma.dto.request;

import lombok.Data;

@Data
public class SignUpRequest {
    String firstName;
    String lastName;
    Integer age;
    String dateOfBirth;
    String email;
    String password;
}

