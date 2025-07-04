// import React, { useState, useContext } from "react";
// import { useNavigate } from "react-router-dom"; // Убедитесь, что импорт корректный
// import { AuthContext } from "../AuthProvider"; // Импорт AuthContext
// import "./SignIn.css";
//
// export const SignIn = () => {
//     const [email, setEmail] = useState("");
//     const [password, setPassword] = useState("");
//     const [errorMessage, setErrorMessage] = useState("");
//     const navigate = useNavigate(); // Хук для навигации
//     const { setToken } = useContext(AuthContext); // Получение setToken из AuthContext
//
//     const handleSubmit = async (e) => {
//         e.preventDefault();
//         setErrorMessage("");
//         try {
//             const response = await fetch(`${process.env.REACT_APP_API_URL}/auth/sign-in`, {
//                 method: "POST",
//                 headers: {
//                     "Content-Type": "application/json",
//                 },
//                 body: JSON.stringify({ email, password }),
//             });
//             if (response.ok) {
//                 const data = await response.json();
//                 console.log("Успешный вход:", data);
//
//                 // Сохранение токена в localStorage и обновление AuthContext
//                 localStorage.setItem("token", data.token);
//                 setToken(data.token);
//
//                 navigate("/"); // Перенаправление на главную страницу
//             } else {
//                 setErrorMessage("Не удалось войти. Проверьте email и пароль.");
//             }
//         } catch (error) {
//             setErrorMessage("Произошла ошибка. Попробуйте позже.");
//             console.error("Ошибка:", error);
//         }
//     };
//
//     return (
//         <form onSubmit={handleSubmit}>
//             {errorMessage && <p className="error-message">{errorMessage}</p>}
//             <div>
//                 <label>Email:</label>
//                 <input
//                     type="email"
//                     value={email}
//                     onChange={(e) => setEmail(e.target.value)}
//                 />
//             </div>
//             <div>
//                 <label>Пароль:</label>
//                 <input
//                     type="password"
//                     value={password}
//                     onChange={(e) => setPassword(e.target.value)}
//                 />
//             </div>
//             <button type="submit">Войти</button>
//         </form>
//     );
// };

import React, { useState, useContext } from "react";
import { useNavigate } from "react-router-dom";
import { AuthContext } from "../AuthProvider/AuthProvider"; // Убедитесь, что путь правильный
import "./SignIn.css";

export const SignIn = () => {
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [errorMessage, setErrorMessage] = useState("");
    const navigate = useNavigate();
    const { login } = useContext(AuthContext); // Получаем функцию login из контекста

    const handleSubmit = async (e) => {
        e.preventDefault();
        setErrorMessage("");
        try {
            // Для отладки
            console.log("Отправка запроса:", { email, password });

            const response = await fetch("http://localhost:8081/auth/sign-in", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ email, password }),
            });

            console.log("Статус ответа:", response.status);

            if (response.ok) {
                // Получаем текст ответа и пытаемся его распарсить
                const responseText = await response.text();
                console.log("Ответ текстом:", responseText);

                let data;
                if (responseText) {
                    try {
                        data = JSON.parse(responseText);
                        console.log("Данные после парсинга:", data);
                    } catch (jsonError) {
                        console.error("Ошибка парсинга JSON:", jsonError);
                        throw new Error("Получен некорректный ответ от сервера");
                    }
                } else {
                    throw new Error("Получен пустой ответ от сервера");
                }

                if (data && data.token) {
                    console.log("Успешный вход:", data);
                    login(data.token); // Используем функцию login из AuthContext
                    navigate("/"); // Перенаправление на главную страницу
                } else {
                    setErrorMessage("Не удалось получить токен авторизации");
                    console.error("Структура ответа:", data);
                }
            } else {
                setErrorMessage("Не удалось войти. Проверьте email и пароль.");
            }
        } catch (error) {
            setErrorMessage("Произошла ошибка. Попробуйте позже.");
            console.error("Ошибка:", error);
        }
    };

    return (
        <>
            <form onSubmit={handleSubmit}>
                <h2 className="sign-in-title">Вход в аккаунт</h2>
                {errorMessage && <p className="error-message">{errorMessage}</p>}
                <div>
                    <label>Email:</label>
                    <input
                        type="email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                    />
                </div>
                <div>
                    <label>Пароль:</label>
                    <input
                        type="password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                    />
                </div>
                <button type="submit">Войти</button>
            </form>
        </>
    );
};