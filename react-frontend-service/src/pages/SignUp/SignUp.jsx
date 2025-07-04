import React, { useState, useContext } from "react";
import { useNavigate } from "react-router-dom";
import { AuthContext } from "../AuthProvider/AuthProvider"; // Импорт AuthContext
import "./SignUp.css";

export const SignUp = () => {
    const [formData, setFormData] = useState({
        firstName: "",
        lastName: "",
        age: "",
        dateOfBirth: "",
        email: "",
        password: "",
    });

    const [errorMessage, setErrorMessage] = useState(""); // Состояние для ошибки
    const navigate = useNavigate();
    const { login } = useContext(AuthContext);

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData({ ...formData, [name]: value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setErrorMessage(""); // Сброс сообщения об ошибке перед запросом
        try {
            const formattedData = {
                ...formData,
                age: parseInt(formData.age, 10),
            };

            const dateOfBirth = new Date(formattedData.dateOfBirth);
            if (dateOfBirth > new Date()) {
                setErrorMessage("Дата рождения не может быть в будущем");
                return;
            }

            const response = await fetch(`${process.env.REACT_APP_API_URL}/auth/sign-up`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(formattedData),
            });

            if (response.ok) {
                const data = await response.json();
                console.log("Успешная регистрация:", data);

                // Сохранение токена в localStorage и обновление AuthContext
                login(data.token); // Используем функцию login из AuthContext

                navigate("/"); // Перенаправление на главную страницу
            } else {
                setErrorMessage("Не удалось зарегистрировать пользователя. Попробуйте снова.");
            }
        } catch (error) {
            setErrorMessage("Произошла ошибка. Попробуйте позже.");
            console.error("Ошибка:", error);
        }
    };

    return (
        <form onSubmit={handleSubmit}>
            <h2 className="sign-in-title">Регистрация</h2>
            {errorMessage && <p className="error-message">{errorMessage}</p>} {/* Отображение ошибки */}
            <div>
                <label>Имя:</label>
                <input
                    type="text"
                    name="firstName"
                    value={formData.firstName}
                    onChange={handleChange}
                />
            </div>
            <div>
                <label>Фамилия:</label>
                <input
                    type="text"
                    name="lastName"
                    value={formData.lastName}
                    onChange={handleChange}
                />
            </div>
            <div>
                <label>Возраст:</label>
                <input
                    type="number"
                    name="age"
                    value={formData.age}
                    onChange={handleChange}
                />
            </div>
            <div>
                <label>Дата рождения:</label>
                <input
                    type="date"
                    name="dateOfBirth"
                    value={formData.dateOfBirth}
                    onChange={handleChange}
                />
            </div>
            <div>
                <label>Email:</label>
                <input
                    type="email"
                    name="email"
                    value={formData.email}
                    onChange={handleChange}
                />
            </div>
            <div>
                <label>Пароль:</label>
                <input
                    type="password"
                    name="password"
                    value={formData.password}
                    onChange={handleChange}
                />
            </div>
            <button type="submit">Зарегистрироваться</button>
        </form>
    );
};