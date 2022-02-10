Уравнение теплопроводности в полярных координатах:

$$
\partial_t U = \frac{\partial_r(r\cdot \partial_r U)}{r} + \frac{\partial_\varphi^2 U}{r^2} \\

\partial_t U = \frac{\partial_r U}{r} + \partial_r^2 U + \frac{\partial_\varphi^2 U}{r^2}
$$

Заменим поизводные конечными разностями:

$$
\frac{U_{i, j}^{k+1} - U_{i, j}^{k}}{\Delta t_k} = \frac{U_{i + 1, j}^{k} - U_{i, j}^{k}}{\Delta r_i \cdot r_i}  
+ \frac{U_{i + 1, j}^{k} - 2 U_{i, j}^{k} + U_{i - 1, j}^{k}}{{\Delta r_i}^2} 
+ \frac{U_{i, j + 1}^{k} - 2U_{i, j}^{k} + U_{i, j- 1}^{k}}{(r_i \cdot \Delta \varphi_j)^2}
$$

$$
    U_{i, j}^{k + 1} = \Delta t_k \cdot \left(
        \frac{U_{i + 1, j}^{k} - U_{i, j}^{k}}{\Delta r_i \cdot r_i}  
        + \frac{U_{i + 1, j}^{k} - 2 U_{i, j}^{k} + U_{i - 1, j}^{k}}{{\Delta r_i}^2} 
        + \frac{U_{i, j + 1}^{k} - 2U_{i, j}^{k} + U_{i, j- 1}^{k}}{(r_i \cdot \Delta \varphi_j)^2}
    \right) + U_{i, j}^{k}
$$