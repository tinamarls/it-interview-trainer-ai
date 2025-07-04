// Header.jsx
import { memo, useContext } from 'react';
import { Button, Container, Navbar, Nav } from 'react-bootstrap';
import { useNavigate } from 'react-router-dom';
import { AuthContext } from '../pages/AuthProvider';
import './Header.css';

export const Header = memo(() => {
    const navigate = useNavigate();
    const { token, logout } = useContext(AuthContext);

    const handleLogout = () => {
        logout(() => navigate('/sign-in')); // навигация безопасно вызывается здесь
    };

    return (
        <Navbar bg="light" expand="lg" fixed="top" className="custom-navbar">
            <Container>

                <Navbar.Toggle aria-controls="basic-navbar-nav" />
                <Navbar.Collapse id="basic-navbar-nav">
                    {token ? (
                        <Nav className="me-auto">
                            <Nav.Link onClick={() => navigate('/resume')}>Мои резюме</Nav.Link>
                            <Nav.Link onClick={() => navigate('/vacancy')}>Анализ вакансии</Nav.Link>
                            <Nav.Link onClick={() => navigate('/compare')}>Сравнение резюме и вакансии</Nav.Link>
                            {/*<Nav.Link onClick={() => navigate('/interview-calm')}>Тренажер спокойствия</Nav.Link>*/}
                            <Nav.Link onClick={() => navigate('/hr-interview')}>HR-скрининг</Nav.Link>
                            <Nav.Link onClick={() => navigate('/hr-interview/results')}>Результаты HR-скрининга</Nav.Link>
                        </Nav>
                    ) : (
                        <Nav className="me-auto" />
                    )}
                    <Nav>
                        {!token ? (
                            <>
                                <Button
                                    variant="outline-primary"
                                    size="sm"
                                    className="me-2"
                                    onClick={() => navigate('/sign-in')}
                                >
                                    Войти
                                </Button>
                                <Button
                                    variant="outline-primary"
                                    size="sm"
                                    onClick={() => navigate('/sign-up')}
                                >
                                    Регистрация
                                </Button>
                            </>
                        ) : (
                            <Button
                                variant="outline-danger"
                                size="sm"
                                onClick={handleLogout}
                            >
                                Выйти
                            </Button>
                        )}
                    </Nav>
                </Navbar.Collapse>
            </Container>
        </Navbar>
    );
});
